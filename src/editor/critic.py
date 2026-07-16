"""The three C4 feedback critics + the complaint->instruction distiller.

The loop varies ONLY the critic; everything else (editor, objective, drift) is
held fixed. The critics reuse the frozen Qwen2-VL judge and the PARA persona /
prompt / parsing machinery already in the repo — nothing about the VLM is
retrained.

- ``StaticCritic``    : no model call; a fixed "improve this image" string.
- ``BlindVLMCritic``  : one generic (no-persona) critique per step.
- ``SocietyCritic``   : N persona critiques per step, aggregated (the method).

``distill_instruction`` turns the running instruction + the newly aggregated
complaints into ONE <=15-word imperative edit, so complaints accumulate across
steps while the editor keeps re-editing the original image (anchored re-edit).

This module is torch-free at import (``para_pipeline`` pulls only pandas/numpy);
the VLM ``backend`` is injected by the caller.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Reuse the PARA persona/prompt/parsing builders (torch-free). They live in
# script/, which is not a package, so put it on sys.path and import the module.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "script"))
import para_pipeline as para  # noqa: E402  (build_para_description, prompts, parser, ScoreDimension)

# A 0-100 "score" axis so we can reuse para.parse_para_rating for the critics'
# JSON ({"score":..,"comment":..}) exactly like the persona runs parse ratings.
_SCORE_DIM = para.ScoreDimension("score", 0.0, 100.0, 1.0, "overall appeal")

_BLIND_QUESTION = (
    "Look at this image. Give it an overall aesthetic score from 0 to 100, then "
    "name the SINGLE most important change that would most improve it. Respond "
    "with ONLY a JSON object and nothing else, in exactly this form:\n"
    '{"score": <integer 0-100>, "comment": "<one concrete improvement, imperative>"}'
)

_PERSONA_QUESTION = (
    "As this exact person, look at this image. Give it an overall aesthetic score "
    "from 0 to 100 for how much YOU personally like it, then name the SINGLE change "
    "that would most improve it for your taste. Respond with ONLY a JSON object and "
    "nothing else, in exactly this form:\n"
    '{"score": <integer 0-100>, "comment": "<one concrete improvement, imperative>"}'
)

_DISTILLER_SYSTEM = (
    "You are a photo-editing assistant. You convert viewer feedback into ONE "
    "concrete, imperative image-edit instruction. Output only the instruction — "
    "no preamble, no quotes, no explanation."
)


@dataclass
class Critique:
    """One step's aggregated feedback."""

    complaints: List[str] = field(default_factory=list)  # one per panelist / one / none
    panel_scores: List[float] = field(default_factory=list)  # 0-100 per panelist
    raw: List[str] = field(default_factory=list)


class Critic:
    """Base: return a :class:`Critique` for an image."""

    def critique(self, image) -> Critique:  # pragma: no cover - interface
        raise NotImplementedError


class StaticCritic(Critic):
    """No model — the loop uses a fixed instruction for this condition."""

    FIXED = "Improve the overall aesthetic quality of this image."

    def critique(self, image) -> Critique:
        return Critique()


class BlindVLMCritic(Critic):
    """One generic, no-persona VLM critique per step (the baseline to beat)."""

    def __init__(self, backend, max_new_tokens: int = 96):
        self.backend = backend
        self.max_new_tokens = max_new_tokens

    def critique(self, image) -> Critique:
        raw = self.backend.generate(
            system_prompt=para.PARA_GENERIC_SYSTEM_PROMPT,
            image=image,
            prompt=_BLIND_QUESTION,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        scores, comment = para.parse_para_rating(raw, [_SCORE_DIM])
        return Critique(
            complaints=[comment] if comment else [],
            panel_scores=[scores["score"]] if scores.get("score") is not None else [],
            raw=[raw],
        )


class SocietyCritic(Critic):
    """N persona critiques per step, aggregated (the synthetic-audience method)."""

    def __init__(self, backend, panel: List[str], max_new_tokens: int = 96):
        self.backend = backend
        self.panel = list(panel)  # persona description strings
        self.max_new_tokens = max_new_tokens

    def critique(self, image) -> Critique:
        n = len(self.panel)
        system_prompts = [para.para_system_prompt(desc) for desc in self.panel]
        raws = self.backend.generate_batch(
            system_prompts,
            [image] * n,
            [_PERSONA_QUESTION] * n,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        complaints, panel_scores = [], []
        for raw in raws:
            scores, comment = para.parse_para_rating(raw, [_SCORE_DIM])
            if comment:
                complaints.append(comment)
            if scores.get("score") is not None:
                panel_scores.append(scores["score"])
        return Critique(complaints=complaints, panel_scores=panel_scores, raw=list(raws))


def build_critic(condition: str, backend=None, panel: Optional[List[str]] = None) -> Critic:
    """Factory for the driver's ``--conditions`` list.

    ``static`` needs no backend; ``blind`` needs the backend; ``society`` needs
    the backend + a persona panel. ``reward_only`` has no critic (handled by the
    loop), so it maps to a :class:`StaticCritic` placeholder.
    """
    if condition in ("static", "reward_only"):
        return StaticCritic()
    if condition == "blind":
        return BlindVLMCritic(backend)
    if condition == "society":
        if not panel:
            raise ValueError("society critic requires a non-empty persona panel")
        return SocietyCritic(backend, panel)
    raise ValueError(f"unknown condition {condition!r}")


# --------------------------------------------------------------------------
# Complaint -> single edit instruction
# --------------------------------------------------------------------------

def _clean_instruction(text: str, max_words: int) -> str:
    """Strip fences/quotes/labels and cap to ``max_words`` words."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    # take the first line, drop a leading "Instruction:" style label
    t = t.splitlines()[0].strip() if t else t
    for label in ("instruction:", "edit:", "output:"):
        if t.lower().startswith(label):
            t = t[len(label):].strip()
    t = t.strip().strip('"').strip("'").strip()
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    return t


def distill_instruction(
    backend,
    image,
    accumulated: str,
    new_complaints: List[str],
    *,
    max_words: int = 15,
    max_new_tokens: int = 48,
) -> str:
    """Merge the running instruction + new complaints into ONE <=15-word edit.

    Accumulation across steps happens here: the previous ``accumulated``
    instruction is passed back in so the distiller grows/merges it rather than
    replacing it, while the editor keeps applying the result to the *original*
    image. If there are no complaints (e.g. the panel is content), the previous
    instruction is kept.
    """
    complaints = [c.strip() for c in new_complaints if c and c.strip()]
    if not complaints:
        return accumulated
    bullets = "\n".join(f"- {c}" for c in complaints[:12])
    prev = (
        f"The current edit instruction is: \"{accumulated}\".\n" if accumulated else ""
    )
    prompt = (
        f"{prev}Viewers looking at this image want these changes:\n{bullets}\n\n"
        f"Rewrite everything into ONE concrete imperative photo-edit instruction of at "
        f"most {max_words} words that an image editor can apply. Output only the instruction."
    )
    raw = backend.generate(
        system_prompt=_DISTILLER_SYSTEM,
        image=image,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    cleaned = _clean_instruction(raw, max_words)
    return cleaned or accumulated
