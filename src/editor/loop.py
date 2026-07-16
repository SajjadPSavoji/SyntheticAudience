"""The C4 auto-refinement loop: anchored re-edit + accept-if-better.

Pure orchestration — no torch/diffusers import here. It receives already-built
``editor`` / ``objective`` / ``drift`` / ``critic`` objects plus a ``distill``
callable, so it stays unit-testable and the heavy pieces are swappable.

Per step (R steps total):
  1. the critic looks at the CURRENT BEST image (society/blind) — static/reward
     conditions skip this;
  2. its complaints are distilled + accumulated into one growing instruction;
  3. the editor re-edits the ORIGINAL source with that instruction (K seeds) —
     "anchored", so artifacts never compound;
  4. each candidate is scored by the held-out objective and the drift guardrail;
  5. among drift-feasible candidates take the objective-max, and COMMIT it only
     if it beats the running best (accept-if-better) — so best-so-far is
     monotone non-decreasing by construction.

Only Pillow is imported (base-env safe).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from PIL import Image

DEFAULT_STATIC_INSTRUCTION = "Improve the overall aesthetic quality of this image."

# Generic edit prompts for the reward-only oracle (no critic guidance). Cycled
# across steps so it explores varied edits and keeps whatever scores highest.
DEFAULT_REWARD_BANK = [
    "Improve the overall aesthetic quality of this image.",
    "Enhance lighting, contrast, and color balance.",
    "Sharpen details and reduce noise.",
    "Improve composition and remove distracting elements.",
    "Make the colors richer and more vibrant.",
    "Brighten the image and improve dynamic range.",
    "Increase clarity and depth of field.",
    "Give the image a cleaner, more professional look.",
]


@dataclass
class StepRecord:
    step: int
    instruction: str
    complaints: List[str] = field(default_factory=list)
    panel_scores: List[float] = field(default_factory=list)
    candidates: List[dict] = field(default_factory=list)  # {path, seed, aesthetic, drift, feasible}
    committed_candidate: Optional[int] = None
    best_obj: float = 0.0
    best_path: str = ""
    drift_of_best: float = 1.0


class EditCache:
    """Disk-backed cache of scored candidates keyed by the generation inputs.

    The generated PNG lives on disk; the index maps a content key to its path +
    objective + drift, so ``--resume`` skips regenerating (FLUX is the cost).
    """

    def __init__(self, path: Optional[str]):
        self.path = path
        self._idx: dict = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self._idx = json.load(f)

    @staticmethod
    def key(image_id, condition, step, instruction, seed, editor_name) -> str:
        h = hashlib.sha1(
            "|".join(str(x) for x in
                     (image_id, condition, step, instruction, seed, editor_name)).encode()
        )
        return h.hexdigest()

    def get(self, key: str) -> Optional[dict]:
        rec = self._idx.get(key)
        if rec and os.path.exists(rec["path"]):
            return rec
        return None

    def put(self, key: str, rec: dict) -> None:
        self._idx[key] = rec
        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._idx, f)
            os.replace(tmp, self.path)


def _save(img: Image.Image, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path


def run_refinement(
    image_id: str,
    source_path: str,
    *,
    condition: str,
    editor,
    objective,
    drift,
    critic,
    distill: Callable[[Image.Image, str, List[str]], str],
    save_dir: str,
    R: int = 10,
    K: int = 3,
    drift_cap: float = 0.85,
    seed: int = 0,
    cache: Optional[EditCache] = None,
    reward_bank: Optional[List[str]] = None,
    static_instruction: str = DEFAULT_STATIC_INSTRUCTION,
) -> List[StepRecord]:
    """Run the loop for one (image, condition). Returns per-step records."""
    reward_bank = reward_bank or DEFAULT_REWARD_BANK
    out_dir = os.path.join(save_dir, condition, str(image_id))
    os.makedirs(out_dir, exist_ok=True)

    # Step 0: the source image is the initial best.
    source_img = Image.open(source_path).convert("RGB")
    src_saved = _save(source_img, os.path.join(out_dir, "step0_source.png"))
    best_obj = objective.score(source_img)
    best_img = source_img
    best_path = src_saved
    accumulated = static_instruction if condition == "static" else ""

    records = [StepRecord(step=0, instruction="", best_obj=best_obj,
                          best_path=best_path, drift_of_best=1.0)]

    for r in range(1, R + 1):
        # 1-2) instruction for this step
        complaints: List[str] = []
        panel_scores: List[float] = []
        if condition == "static":
            instruction = static_instruction
        elif condition == "reward_only":
            instruction = reward_bank[(r - 1) % len(reward_bank)]
        else:  # blind / society: critique the CURRENT BEST, then distill
            crit = critic.critique(best_img)
            complaints, panel_scores = crit.complaints, crit.panel_scores
            instruction = distill(best_img, accumulated, complaints)
            accumulated = instruction

        # 3) anchored re-edit of the ORIGINAL source (K candidates)
        step_seed = seed + 1000 * r
        cand_imgs: List[Image.Image] = []
        cand_meta: List[dict] = []
        to_score: List[int] = []  # indices needing fresh scoring
        for kk in range(K):
            ckey = None if cache is None else EditCache.key(
                image_id, condition, r, instruction, step_seed + kk, editor.name)
            hit = cache.get(ckey) if ckey else None
            if hit:
                cand_imgs.append(Image.open(hit["path"]).convert("RGB"))
                cand_meta.append({"path": hit["path"], "seed": step_seed + kk,
                                  "aesthetic": hit["aesthetic"], "drift": hit["drift"]})
            else:
                cand_imgs.append(None)  # filled after edit
                cand_meta.append({"path": None, "seed": step_seed + kk,
                                  "aesthetic": None, "drift": None, "_key": ckey})
                to_score.append(kk)

        if to_score:
            edited = editor.edit(source_path, instruction, k=K, seed=step_seed)
            objs = objective.score_batch([edited[i] for i in to_score])
            drifts = drift.similarity_batch(source_img, [edited[i] for i in to_score])
            for j, kk in enumerate(to_score):
                img = edited[kk]
                path = _save(img, os.path.join(out_dir, f"step{r}_cand{kk}.png"))
                cand_imgs[kk] = img
                cand_meta[kk].update(path=path, aesthetic=objs[j], drift=drifts[j])
                if cache is not None and cand_meta[kk].get("_key"):
                    cache.put(cand_meta[kk]["_key"],
                              {"path": path, "aesthetic": objs[j], "drift": drifts[j]})
                cand_meta[kk].pop("_key", None)

        # 4) feasibility + 5) accept-if-better
        for m in cand_meta:
            m["feasible"] = bool(m["drift"] is not None and m["drift"] >= drift_cap)
        feasible = [i for i, m in enumerate(cand_meta) if m["feasible"]]
        committed = None
        if feasible:
            best_i = max(feasible, key=lambda i: cand_meta[i]["aesthetic"])
            if cand_meta[best_i]["aesthetic"] > best_obj:
                best_obj = cand_meta[best_i]["aesthetic"]
                best_img = cand_imgs[best_i]
                best_path = _save(best_img, os.path.join(out_dir, f"step{r}_best.png"))
                committed = best_i

        records.append(StepRecord(
            step=r,
            instruction=instruction,
            complaints=complaints,
            panel_scores=panel_scores,
            candidates=[{k: v for k, v in m.items() if not k.startswith("_")} for m in cand_meta],
            committed_candidate=committed,
            best_obj=best_obj,
            best_path=best_path,
            drift_of_best=(cand_meta[committed]["drift"] if committed is not None
                           else records[-1].drift_of_best),
        ))

    return records


def records_to_dicts(image_id: str, condition: str, records: List[StepRecord]) -> List[dict]:
    """Flatten StepRecords into per-step log rows (mirrors data/results schema)."""
    rows = []
    for rec in records:
        row = asdict(rec)
        row["image_id"] = image_id
        row["condition"] = condition
        rows.append(row)
    return rows
