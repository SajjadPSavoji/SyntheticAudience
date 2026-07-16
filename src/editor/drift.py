"""Identity-drift guardrail for the C4 loop.

Cosine similarity between the source image and an edited candidate in DINOv2
embedding space (CLS token). The loop rejects any candidate whose similarity
falls below a cap, which (a) keeps "better" from meaning "drifted into a
different image" and (b) blunts reward-hacking of the aesthetic objective.

Heavy imports are deferred into ``__init__``.
"""
from __future__ import annotations

from typing import List, Sequence

from persona.backend.base import ImageInput, load_image  # Pillow-only, import-safe


class DriftMetric:
    """DINOv2 CLS-token cosine similarity of edited-vs-source images.

    ``similarity`` returns a value in roughly [-1, 1]; identity-preserving edits
    sit near 1.0. A typical acceptance cap is ~0.85.
    """

    def __init__(self, backbone: str = "facebook/dinov2-base", device: str = "cuda", dtype=None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.device = device
        self.dtype = torch.float16 if dtype is None else dtype
        self.processor = AutoImageProcessor.from_pretrained(backbone)
        self.model = AutoModel.from_pretrained(backbone, torch_dtype=self.dtype).to(device)
        self.model.eval()

    def _embed(self, images: Sequence[ImageInput]):
        torch = self._torch
        pil = [load_image(im) for im in images]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**{k: v.to(self.dtype) if v.is_floating_point() else v
                                for k, v in inputs.items()})
            cls = out.last_hidden_state[:, 0]  # CLS token
            cls = cls.float()
            cls = cls / cls.norm(dim=-1, keepdim=True)
        return cls

    def similarity(self, src: ImageInput, edited: ImageInput) -> float:
        return self.similarity_batch(src, [edited])[0]

    def similarity_batch(self, src: ImageInput, edited_list: Sequence[ImageInput]) -> List[float]:
        if not edited_list:
            return []
        embs = self._embed([src, *edited_list])
        src_emb, cand_embs = embs[0:1], embs[1:]
        sims = (cand_embs * src_emb).sum(dim=-1)  # cosine (already L2-normalized)
        return [float(s) for s in sims.cpu()]
