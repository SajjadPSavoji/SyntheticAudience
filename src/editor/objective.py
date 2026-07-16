"""Held-out quality objective for the C4 loop (aesthetic-only, text-free).

The objective is the LAION "improved aesthetic predictor": a small MLP head on
top of L2-normalized CLIP ViT-L/14 image embeddings, trained on AVA/SAC/LOGOS
aesthetic ratings. It needs **no text**, and it is a different model family
from the Qwen2-VL critic — so it satisfies the C4 ``critic != objective`` rule
(the critic proposes edits; this independent model judges whether they helped).

Heavy imports are deferred into ``__init__``. Weights are fetched once and
cached; by default from the authoritative GitHub release, overridable to an HF
repo or a local file for offline / Colab-Drive use.
"""
from __future__ import annotations

import os
import urllib.request
from typing import List, Sequence

from persona.backend.base import ImageInput, load_image  # Pillow-only, import-safe

# Authoritative weights for the ViT-L/14 linear-MSE aesthetic head.
_DEFAULT_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    "raw/main/sac+logos+ava1-l14-linearMSE.pth"
)
_DEFAULT_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"


def _resolve_weights(
    weights_path: str | None, hf_repo_id: str | None, hf_filename: str
) -> str:
    """Return a local path to the aesthetic-head weights, downloading if needed."""
    if weights_path and os.path.exists(weights_path):
        return weights_path
    if hf_repo_id:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "syntheticaudience", "aesthetic"
    )
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, _DEFAULT_FILENAME)
    if not os.path.exists(dest):
        urllib.request.urlretrieve(_DEFAULT_WEIGHTS_URL, dest)
    return dest


class AestheticObjective:
    """Score any image with the LAION improved-aesthetic predictor (~1-10).

    Parameters
    ----------
    clip_model:
        CLIP backbone whose image embeddings feed the head. Must match the head
        (ViT-L/14 -> 768-d).
    weights_path / hf_repo_id / hf_filename:
        Where to get the MLP head weights (local path wins; else HF repo; else
        the default GitHub URL, cached under ~/.cache).
    device, dtype:
        CLIP runs in fp16 on GPU by default; the tiny head runs in fp32.
    """

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-large-patch14",
        weights_path: str | None = None,
        hf_repo_id: str | None = None,
        hf_filename: str = _DEFAULT_FILENAME,
        device: str = "cuda",
        dtype=None,
    ) -> None:
        import torch
        import torch.nn as nn
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device
        self.dtype = torch.float16 if dtype is None else dtype

        self.clip = CLIPModel.from_pretrained(clip_model, torch_dtype=self.dtype).to(device)
        self.clip.eval()
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        embed_dim = self.clip.config.projection_dim  # 768 for ViT-L/14

        # The improved-aesthetic-predictor head architecture.
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        path = _resolve_weights(weights_path, hf_repo_id, hf_filename)
        state = torch.load(path, map_location="cpu")
        # The released checkpoint keys are "layers.N.*"; match by loading directly.
        self.head.load_state_dict({k.replace("layers.", ""): v for k, v in state.items()}
                                  if any(k.startswith("layers.") for k in state)
                                  else state)
        self.head.to(device).float().eval()

    def score(self, image: ImageInput) -> float:
        return self.score_batch([image])[0]

    def score_batch(self, images: Sequence[ImageInput]) -> List[float]:
        torch = self._torch
        pil = [load_image(im) for im in images]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            feats = self.clip.get_image_features(
                pixel_values=inputs["pixel_values"].to(self.dtype)
            ).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalize
            scores = self.head(feats).squeeze(-1)
        return [float(s) for s in scores.cpu()]
