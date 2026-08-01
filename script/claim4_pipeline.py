"""Claim-4 pipeline: iterative persona-guided image editing on PARA images.

This is a *closed feedback loop* between two Qwen models, not a fidelity study:

  1. Sample N PARA images and pin one random PARA annotator (a real rater, with
     their demographics + Big-Five profile) to each image as a fixed persona.
  2. Show the current image to Qwen2-VL role-playing that persona and ask, in
     character, for (a) a 1-5 aesthetic score and (b) one concrete visual edit
     that would most improve the photo *for this person's taste*.
  3. Feed that edit instruction and the current image to Qwen-Image-Edit-2511,
     which returns a new image.
  4. Repeat step 2-3 --n-iterations times (default 10), each edit acting on the
     previous edit's output. A final evaluation is run on the last edited image,
     so an N-edit run has N+1 evaluations (scores for image_0 ... image_N).

The question this answers: if you keep editing a photo toward one person's
stated preferences, does that person's aesthetic score climb, and how does the
image drift over ten rounds?

The persona machinery (description builder, role-play system prompt, aesthetic
1-5 grid) is imported straight from para_pipeline.py so the annotator is
described to the VLM exactly as in the fidelity experiments; only the *question*
is new (it asks for an edit instruction, which PARA never did).

Two GPUs are used by default: the editor (~62 GB peak) on one and the 7B judge
(~16 GB) on another, since they do not fit comfortably together. Override with
--edit-device / --vlm-device (both default to cuda:0 if only one is visible).

Layout written under data/logs_claim4/ (one folder per image):

    data/logs_claim4/
      summary.json                       # config + per-image score trajectories
      <sessionId>__<imageStem>/
        iter_00_initial.jpg              # the (resized) starting image
        iter_01.jpg ... iter_10.jpg      # image after each edit
        log.json                         # persona + per-iteration score/comment/instruction

Run from the repo root (persona conda env), ideally via script/claim4_pipeline.sh
which picks two free GPUs for you:

    python script/claim4_pipeline.py --n-images 10 --n-iterations 10 --seed 0
    python script/claim4_pipeline.py --n-images 1 --n-iterations 2   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

# Make ``src`` importable and pull the PARA persona machinery in directly (same
# annotator description + role-play prompt + aesthetic grid as the fidelity runs).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "script"))

from para_pipeline import (  # noqa: E402
    PARA_ANNOTATION_DIR,
    PARA_DIMENSIONS,
    PARA_IMAGE_DIR,
    ScoreDimension,
    _atomic_write_json,
    build_para_description,
    choose_images,
    para_system_prompt,
)

DEFAULT_LOG_DIR = REPO_ROOT / "data" / "logs_claim4"
DEFAULT_VLM_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_EDIT_MODEL = "Qwen/Qwen-Image-Edit-2511"

# The single axis this experiment tracks over iterations: PARA overall aesthetics
# on its 1-5 half-point grid (reused so scores are directly comparable to PARA).
AESTHETIC = PARA_DIMENSIONS["aestheticScore"]


# --------------------------------------------------------------------------
# Prompt + parsing (asks for a score AND an actionable edit instruction)
# --------------------------------------------------------------------------

def build_claim4_question(dim: ScoreDimension) -> str:
    """Question that elicits a 1-5 score plus one concrete image-edit instruction.

    Unlike para_pipeline's question (score + a comment explaining it), this asks
    the persona for a *directive*: a single, self-contained visual edit an image
    editor could apply. The instruction is fed verbatim to Qwen-Image-Edit, so
    the wording steers the persona toward imperative, visual changes rather than
    meta-commentary about the photo.
    """
    return (
        "As this annotator, react to this photograph as if deciding whether to "
        "like it in your social feed. Then tell an image editor how to change it "
        "so that YOU personally would like it more.\n\n"
        f'- "aestheticScore": your overall aesthetic rating of the photo as it is '
        f"now — {dim.prompt}. Rate from {dim.lo:g} to {dim.hi:g} in steps of "
        f"{dim.step:g}.\n"
        '- "edit_instruction": ONE concrete, visual editing instruction that would '
        "most improve this photo for your taste, phrased as a direct command to an "
        "image editor (e.g. 'brighten the sky and boost the contrast', 'crop "
        "tighter on the subject and blur the background', 'warm the colours and "
        "remove the sign on the left'). Describe only visual changes to this "
        "image; do not mention scores, yourself, or the editor.\n"
        '- "comment": one short in-character sentence on why you want that change.\n\n'
        "Respond with ONLY a single JSON object and nothing else (no markdown, no "
        "extra text), in exactly this form:\n"
        '{"aestheticScore": <number>, "edit_instruction": "<instruction>", '
        '"comment": "<sentence>"}'
    )


_STR_FIELD_RE = {
    "edit_instruction": re.compile(r'"edit_instruction"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    "comment": re.compile(r'"comment"\s*:\s*"((?:[^"\\]|\\.)*)"'),
}


def parse_claim4(
    raw_response: str, dim: ScoreDimension
) -> tuple[Optional[float], str, str]:
    """Parse ``(aestheticScore_or_None, edit_instruction, comment)`` from a reply.

    Strict JSON first (after stripping any markdown fence), then a per-field regex
    fallback for dirty output — same strategy as para_pipeline.parse_para_rating.
    A missing score yields ``None`` (the iteration still runs); a missing
    instruction yields ``""`` (the caller skips the edit for that round).
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end > start else text
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        data = None

    score: Optional[float] = None
    if isinstance(data, dict) and dim.key in data:
        try:
            score = dim.snap(float(data[dim.key]))
        except (TypeError, ValueError):
            score = None
    if score is None:
        m = re.search(rf'"{re.escape(dim.key)}"\s*:\s*"?(-?\d+(?:\.\d+)?)', text)
        score = dim.snap(float(m.group(1))) if m else None

    def _field(name: str) -> str:
        if isinstance(data, dict) and name in data:
            return str(data.get(name, "")).strip()
        m = _STR_FIELD_RE[name].search(text)
        return m.group(1).strip() if m else ""

    return score, _field("edit_instruction"), _field("comment")


# --------------------------------------------------------------------------
# Qwen-Image-Edit wrapper
# --------------------------------------------------------------------------

class QwenImageEditor:
    """Thin wrapper over diffusers' QwenImageEditPlusPipeline.

    The cached Qwen-Image-Edit-2511 snapshot on this machine is missing its
    ``scheduler/`` subfolder, and the shared HF cache is read-only, so the
    pipeline's own attempt to fetch the scheduler fails with a PermissionError.
    We fetch just the (tiny) scheduler config into a writable cache and pass it
    explicitly, which also stops the pipeline touching the shared cache.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EDIT_MODEL,
        device: str = "cuda:0",
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
    ) -> None:
        import torch
        from diffusers import (
            FlowMatchEulerDiscreteScheduler,
            QwenImageEditPlusPipeline,
        )

        self.device = device
        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale
        self.negative_prompt = negative_prompt

        writable_cache = os.path.expanduser("~/.cache/huggingface/hub")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler", cache_dir=writable_cache
        )
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_name, scheduler=scheduler, torch_dtype=torch.bfloat16
        ).to(device)

    def edit(self, image: Image.Image, instruction: str, seed: int) -> Image.Image:
        import torch

        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            image=[image],
            prompt=instruction,
            negative_prompt=self.negative_prompt,
            true_cfg_scale=self.true_cfg_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        )
        return out.images[0]


def edit_with_retry(
    editor: QwenImageEditor,
    image: Image.Image,
    instruction: str,
    seed: int,
    attempts: int = 3,
    backoff: float = 8.0,
) -> Image.Image:
    """Run one edit, surviving the transient CUDA failures of a shared GPU.

    Mirrors para_pipeline.generate_with_retry: a failed attempt releases our own
    cached blocks (retrying while the allocator still holds the starved memory is
    pointless) and waits, doubling, before trying again."""
    import torch

    for i in range(1, attempts + 1):
        try:
            return editor.edit(image, instruction, seed)
        except Exception as exc:  # noqa: BLE001 — transient CUDA faults, retry
            torch.cuda.empty_cache()
            if i == attempts:
                raise
            wait = backoff * 2 ** (i - 1)
            print(
                f"    edit failed ({type(exc).__name__}: {exc}); "
                f"attempt {i}/{attempts}, retrying in {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------
# Image + persona selection
# --------------------------------------------------------------------------

def _resize_max(img: Image.Image, max_size: int) -> Image.Image:
    """Downscale so the longest side is <= max_size (keeps memory/time bounded
    and the working resolution stable across iterations). No upscaling."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_size:
        return img
    scale = max_size / longest
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def _folder_name(session_id: str, image_name: str) -> str:
    """Collision-proof per-image folder name: ``<sessionId>__<imageStem>``."""
    return f"{session_id}__{Path(image_name).stem}"


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Iteratively edit PARA images toward one random persona's stated "
            "preferences, using Qwen2-VL as the critic and Qwen-Image-Edit-2511 "
            "as the editor."
        )
    )
    p.add_argument("--n-images", type=int, default=10, help="PARA images to run (default: %(default)s).")
    p.add_argument("--n-iterations", type=int, default=10, help="Edit rounds per image (default: %(default)s).")
    p.add_argument(
        "--images", default=None,
        help="Comma-separated imageName values to run instead of sampling.",
    )
    p.add_argument(
        "--sampling", choices=["stratified", "uniform"], default="stratified",
        help="Image sampling: 'stratified' spreads over the human mean-score range "
        "(default: %(default)s).",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for image + persona sampling (default: 0).")
    p.add_argument("--edit-seed", type=int, default=0, help="Base seed for the diffusion generator (default: 0).")
    p.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL, help="HF id of the critic VLM (default: %(default)s).")
    p.add_argument("--edit-model", default=DEFAULT_EDIT_MODEL, help="HF id of the image editor (default: %(default)s).")
    p.add_argument("--vlm-device", default=None, help="Device for the critic VLM (default: cuda:1 if 2+ GPUs else cuda:0).")
    p.add_argument("--edit-device", default=None, help="Device for the image editor (default: cuda:0).")
    p.add_argument("--max-size", type=int, default=1024, help="Longest-side cap for the working image (default: %(default)s).")
    p.add_argument("--num-inference-steps", type=int, default=40, help="Denoising steps per edit (default: %(default)s).")
    p.add_argument("--true-cfg-scale", type=float, default=4.0, help="Classifier-free guidance scale for the editor (default: %(default)s).")
    p.add_argument("--negative-prompt", default=" ", help="Editor negative prompt; a non-empty value enables CFG (default: ' ').")
    p.add_argument("--temperature", type=float, default=0.0, help="Critic sampling temperature; 0 = greedy (default: 0).")
    p.add_argument("--max-new-tokens", type=int, default=160, help="Critic generation budget per evaluation (default: %(default)s).")
    p.add_argument("--output-dir", default=None, help="Output root (default: data/logs_claim4).")
    p.add_argument("--overwrite", action="store_true", help="Re-run images whose folder already has a completed log.json.")
    return p.parse_args()


def _resolve_devices(args: argparse.Namespace) -> tuple[str, str]:
    """Pick (vlm_device, edit_device): split across two GPUs when available."""
    import torch

    n = torch.cuda.device_count()
    edit_device = args.edit_device or "cuda:0"
    if args.vlm_device:
        vlm_device = args.vlm_device
    else:
        vlm_device = "cuda:1" if n >= 2 else "cuda:0"
    return vlm_device, edit_device


def _select_images(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return ``(sessionId, imageName)`` pairs to run."""
    images_df = pd.read_csv(
        PARA_ANNOTATION_DIR / "PARA-Images.csv",
        usecols=["sessionId", "imageName", "aestheticScore"],
    )
    # imageName -> sessionId is 1:1 in PARA (an image lives under one session).
    session_of = (
        images_df.drop_duplicates("imageName").set_index("imageName")["sessionId"].to_dict()
    )
    if args.images:
        names = [n.strip() for n in args.images.split(",") if n.strip()]
        missing = [n for n in names if n not in session_of]
        if missing:
            raise ValueError(f"imageName(s) not in PARA-Images.csv: {missing}")
    else:
        names = choose_images(
            images_df, args.n_images, args.seed, args.sampling, "aestheticScore"
        )
    return [(str(session_of[n]), str(n)) for n in names]


def _assign_personas(
    n: int, users_df: pd.DataFrame, seed: int
) -> list[str]:
    """Pick one random PARA annotator (userId) per image, without replacement
    where possible so distinct images get distinct personas."""
    rng = np.random.default_rng(seed)
    user_ids = users_df.index.to_numpy()
    replace = n > len(user_ids)
    chosen = rng.choice(user_ids, size=n, replace=replace)
    return [str(u) for u in chosen]


def _already_done(folder: Path, n_iterations: int) -> bool:
    """True if this image's folder holds a log.json with all N+1 evaluations."""
    log_path = folder / "log.json"
    if not log_path.exists():
        return False
    try:
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        return len(log.get("iterations", [])) >= n_iterations + 1
    except (json.JSONDecodeError, OSError):
        return False


def _build_vlm_gen_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        kwargs.update(do_sample=True, temperature=args.temperature)
    else:
        kwargs["do_sample"] = False
    return kwargs


def run(args: argparse.Namespace) -> None:
    question = build_claim4_question(AESTHETIC)
    out_root = Path(args.output_dir) if args.output_dir else DEFAULT_LOG_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    users_df = pd.read_csv(PARA_ANNOTATION_DIR / "PARA-UserInfo.csv").set_index("userId")
    images = _select_images(args)
    persona_ids = _assign_personas(len(images), users_df, args.seed)

    print(
        f"Plan: {len(images)} images x {args.n_iterations} edit rounds "
        f"= {len(images) * args.n_iterations} edits, "
        f"{len(images) * (args.n_iterations + 1)} evaluations.\n"
        f"Output root: {out_root}"
    )

    vlm_device, edit_device = _resolve_devices(args)
    print(f"Loading critic VLM '{args.vlm_model}' on {vlm_device} ...")
    from persona import QwenVLBackend  # deferred: pulls in torch

    backend = QwenVLBackend(model_name=args.vlm_model, device_map={"": vlm_device})
    vlm_gen_kwargs = _build_vlm_gen_kwargs(args)

    print(f"Loading image editor '{args.edit_model}' on {edit_device} ...")
    editor = QwenImageEditor(
        model_name=args.edit_model,
        device=edit_device,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt=args.negative_prompt,
    )

    config = {
        "dataset": "PARA",
        "vlm_model": args.vlm_model,
        "edit_model": args.edit_model,
        "n_images": len(images),
        "n_iterations": args.n_iterations,
        "sampling": args.sampling if not args.images else "explicit",
        "seed": args.seed,
        "edit_seed": args.edit_seed,
        "max_size": args.max_size,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "question": question,
    }
    summary_path = out_root / "summary.json"
    summary_images: list[dict] = []
    started = time.time()

    for idx, ((session_id, image_name), user_id) in enumerate(zip(images, persona_ids)):
        folder = out_root / _folder_name(session_id, image_name)
        src_path = PARA_IMAGE_DIR / session_id / image_name
        tag = f"[{idx + 1}/{len(images)}] {session_id}/{image_name}"

        if not src_path.exists():
            print(f"{tag}: image file missing at {src_path}, skipping.", flush=True)
            continue
        if _already_done(folder, args.n_iterations) and not args.overwrite:
            print(f"{tag}: already complete, skipping (use --overwrite to redo).", flush=True)
            with open(folder / "log.json", encoding="utf-8") as f:
                summary_images.append(_summary_row(folder.name, session_id, image_name, json.load(f)))
            continue

        folder.mkdir(parents=True, exist_ok=True)
        description = build_para_description(users_df.loc[user_id])
        system_prompt = para_system_prompt(description)
        print(f"{tag}  persona={user_id}", flush=True)

        working = _resize_max(load_rgb(src_path), args.max_size)
        working.save(folder / "iter_00_initial.jpg", quality=95)

        image_log = {
            "dataset": "PARA",
            "image": {"sessionId": session_id, "imageName": image_name, "source": str(src_path)},
            "persona": {"userId": str(user_id), "description": description},
            "vlm_model": args.vlm_model,
            "edit_model": args.edit_model,
            "config": config,
            "iterations": [],
        }

        for i in range(args.n_iterations + 1):
            image_file = "iter_00_initial.jpg" if i == 0 else f"iter_{i:02d}.jpg"
            raw = backend.generate(system_prompt, working, question, **vlm_gen_kwargs)
            score, instruction, comment = parse_claim4(raw, AESTHETIC)

            record = {
                "iteration": i,
                "image_file": image_file,
                "aestheticScore": score,
                "edit_instruction": instruction,
                "comment": comment,
                "raw_response": raw,
                "edited_to": None,
                "edit_error": None,
            }

            # Every round but the last produces the next image from this critique.
            if i < args.n_iterations:
                if not instruction:
                    record["edit_error"] = "no edit_instruction parsed; image left unchanged"
                    next_file = f"iter_{i + 1:02d}.jpg"
                    working.save(folder / next_file, quality=95)
                    record["edited_to"] = next_file
                else:
                    try:
                        working = edit_with_retry(
                            editor, working, instruction, args.edit_seed + i
                        )
                        next_file = f"iter_{i + 1:02d}.jpg"
                        working.save(folder / next_file, quality=95)
                        record["edited_to"] = next_file
                    except Exception as exc:  # noqa: BLE001
                        record["edit_error"] = f"{type(exc).__name__}: {exc}"
                        image_log["iterations"].append(record)
                        print(f"    iter {i}: edit failed permanently ({exc}); "
                              "stopping this image.", flush=True)
                        break

            image_log["iterations"].append(record)
            _atomic_write_json(folder / "log.json", image_log)  # checkpoint per round
            sc = f"{score:.1f}" if score is not None else "??"
            preview = (instruction[:70] + "…") if len(instruction) > 70 else instruction
            print(f"    iter {i:>2}: score {sc}  edit: {preview}", flush=True)

        _atomic_write_json(folder / "log.json", image_log)
        row = _summary_row(folder.name, session_id, image_name, image_log)
        summary_images.append(row)
        _atomic_write_json(summary_path, {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "images": summary_images,
        })
        traj = row["score_trajectory"]
        print(f"    trajectory: {traj}  (Δ {row['score_delta']})", flush=True)

    _atomic_write_json(summary_path, {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "images": summary_images,
    })
    print(
        f"\nDone: {len(summary_images)} images in {(time.time() - started) / 60:.1f} min. "
        f"Logs under {out_root}"
    )


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _summary_row(folder: str, session_id: str, image_name: str, image_log: dict) -> dict:
    traj = [it["aestheticScore"] for it in image_log.get("iterations", [])]
    scored = [s for s in traj if s is not None]
    delta = (
        round(scored[-1] - scored[0], 2)
        if len(scored) >= 2 else None
    )
    return {
        "folder": folder,
        "sessionId": session_id,
        "imageName": image_name,
        "persona_userId": image_log.get("persona", {}).get("userId"),
        "n_iterations": len(traj) - 1 if traj else 0,
        "score_trajectory": traj,
        "score_start": scored[0] if scored else None,
        "score_end": scored[-1] if scored else None,
        "score_delta": delta,
    }


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
