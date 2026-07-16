"""Auto-research image-editing loop (Claim C4).

A society / agentic simulation is used as the *critic* that drives a 10-step
auto-refinement editing loop; the claim is that this beats a single blind VLM
critic or a fixed instruction. See ``research_plan.md`` §7 / §8.4 / §14.19.

**Deferred-safe:** importing ``editor`` pulls in NO heavy dependencies
(torch / diffusers / transformers). The submodules that need them are loaded
lazily on first attribute access (PEP 562), so the base analysis environment
— which has no GPU stack — can ``import editor`` without failing. This mirrors
the repo convention of deferring ``from persona import QwenVLBackend`` until a
GPU is actually needed (see ``script/eva_pipeline.py``).
"""
from __future__ import annotations

from importlib import import_module

# public symbol -> submodule it lives in (both under the top-level ``editor``
# package, since ``src/`` is on sys.path).
_LAZY = {
    "ImageEditor": "editor.flux_editor",
    "FluxKontextEditor": "editor.flux_editor",
    "InstructPix2PixEditor": "editor.flux_editor",
    "build_editor": "editor.flux_editor",
    "AestheticObjective": "editor.objective",
    "DriftMetric": "editor.drift",
    "Critique": "editor.critic",
    "StaticCritic": "editor.critic",
    "BlindVLMCritic": "editor.critic",
    "SocietyCritic": "editor.critic",
    "build_critic": "editor.critic",
    "distill_instruction": "editor.critic",
    "run_refinement": "editor.loop",
    "StepRecord": "editor.loop",
    "EditCache": "editor.loop",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):  # PEP 562 module-level lazy attribute access
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'editor' has no attribute {name!r}")
    return getattr(import_module(target), name)


def __dir__():
    return __all__
