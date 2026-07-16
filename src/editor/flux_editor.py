"""Instruction image editors for the C4 auto-refinement loop.

The default editor is FLUX.1-Kontext-dev (a diffusers ``FluxKontextPipeline``),
run locally in bf16 — it fits comfortably on an A100-40GB with images on GPU.
A lighter InstructPix2Pix fallback is provided for smaller GPUs (L4/T4).

All heavy imports (torch / diffusers) are **deferred into ``__init__``** so that
``import editor`` stays dependency-free in the base analysis environment.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from persona.backend.base import ImageInput, load_image  # Pillow-only, import-safe


class ImageEditor(ABC):
    """Turn one (image, instruction) into K candidate edits with fixed seeds."""

    name: str = "base"

    @abstractmethod
    def edit(self, image: ImageInput, instruction: str, *, k: int, seed: int) -> List:
        """Return ``k`` edited PIL images. Candidate ``i`` uses ``seed + i`` so the
        whole set is deterministic and cacheable."""
        raise NotImplementedError


class FluxKontextEditor(ImageEditor):
    """FLUX.1-Kontext-dev instruction editor.

    Parameters
    ----------
    model_name:
        Gated HF repo id; the account/token must have accepted its license.
    dtype:
        ``bfloat16`` is the right default on Ampere/Hopper (A100/L4).
    guidance_scale, num_inference_steps:
        Kontext editing defaults (2.5 / 28) — a good speed/fidelity balance.
    cpu_offload:
        ``False`` on A100 (weights live on GPU). Set ``True`` on ~16-24GB cards
        to stream weights via ``enable_model_cpu_offload`` (slower but fits).
    max_side:
        Longest source-image side; larger inputs are downscaled before editing
        to bound VRAM and keep the editor in its trained resolution band.
    """

    name = "flux"

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-Kontext-dev",
        dtype=None,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 28,
        cpu_offload: bool = False,
        max_side: int = 1024,
    ) -> None:
        import torch
        from diffusers import FluxKontextPipeline

        self.model_name = model_name
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_side = max_side
        dtype = torch.bfloat16 if dtype is None else dtype

        self.pipe = FluxKontextPipeline.from_pretrained(model_name, torch_dtype=dtype)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def _prep(self, image: ImageInput):
        img = load_image(image)
        w, h = img.size
        m = max(w, h)
        if m > self.max_side:
            scale = self.max_side / m
            img = img.resize((round(w * scale), round(h * scale)))
        return img

    def edit(self, image: ImageInput, instruction: str, *, k: int, seed: int) -> List:
        import torch

        img = self._prep(image)
        out = []
        for i in range(k):
            gen = torch.Generator("cpu").manual_seed(seed + i)
            result = self.pipe(
                image=img,
                prompt=instruction,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=gen,
            )
            out.append(result.images[0])
        return out


class InstructPix2PixEditor(ImageEditor):
    """Lighter ungated fallback editor (fits ~T4/L4). Not used on A100 runs."""

    name = "instructpix2pix"

    def __init__(
        self,
        model_name: str = "timbrooks/instruct-pix2pix",
        dtype=None,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 20,
        max_side: int = 768,
    ) -> None:
        import torch
        from diffusers import StableDiffusionInstructPix2PixPipeline

        self.model_name = model_name
        self.image_guidance_scale = image_guidance_scale
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_side = max_side
        dtype = torch.float16 if dtype is None else dtype

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_name, torch_dtype=dtype, safety_checker=None
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def _prep(self, image: ImageInput):
        img = load_image(image)
        w, h = img.size
        m = max(w, h)
        if m > self.max_side:
            scale = self.max_side / m
            img = img.resize((round(w * scale), round(h * scale)))
        return img

    def edit(self, image: ImageInput, instruction: str, *, k: int, seed: int) -> List:
        import torch

        img = self._prep(image)
        out = []
        for i in range(k):
            gen = torch.Generator("cpu").manual_seed(seed + i)
            result = self.pipe(
                instruction,
                image=img,
                image_guidance_scale=self.image_guidance_scale,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=gen,
            )
            out.append(result.images[0])
        return out


_EDITORS = {
    "flux": FluxKontextEditor,
    "instructpix2pix": InstructPix2PixEditor,
}


def build_editor(name: str, **kwargs) -> ImageEditor:
    """Factory for the driver's ``--editor`` flag."""
    try:
        cls = _EDITORS[name]
    except KeyError:
        raise ValueError(f"unknown editor {name!r}; choose from {list(_EDITORS)}")
    return cls(**kwargs)
