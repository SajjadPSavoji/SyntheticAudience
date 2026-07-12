"""Qwen2-VL backend used for persona reasoning over images."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .base import ImageInput, VLMBackend, load_image

# TF32 matmuls are a free ~1.5x on Ampere/Hopper and don't measurably change the
# generated text at bf16; enable once at import time.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class QwenVLBackend(VLMBackend):
    """Qwen2-VL backend.

    Parameters
    ----------
    model_name:
        Hugging Face repo id of the model to load. Defaults to the Qwen VLM
        version 2 7B instruct checkpoint. Swap in e.g.
        ``"Qwen/Qwen2.5-VL-7B-Instruct"`` to use the refined 2.5 series.
    device_map:
        Passed to ``from_pretrained``; ``"auto"`` lets Accelerate place the
        weights (a 7B model fits comfortably on a single GPU).
    dtype:
        Compute dtype for the weights. ``bfloat16`` is a good default on
        Ampere/Hopper GPUs.
    max_new_tokens:
        Default generation length for :meth:`generate`.
    attn_implementation:
        Attention kernel passed to ``from_pretrained``; ``"sdpa"`` (PyTorch's
        fused scaled-dot-product attention) is a safe, fast default. Use
        ``"flash_attention_2"`` if flash-attn is installed.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
        attn_implementation: str = "sdpa",
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Batched decoder-only generation requires left padding so that every
        # sequence's newly generated tokens start at the same column.
        self.processor.tokenizer.padding_side = "left"

    @torch.inference_mode()
    def generate(
        self,
        system_prompt: str,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> str:
        return self.generate_batch(
            [system_prompt], [image], [prompt], max_new_tokens, **generate_kwargs
        )[0]

    @torch.inference_mode()
    def generate_batch(
        self,
        system_prompts: Sequence[str],
        images: Sequence[ImageInput],
        prompts: Sequence[str],
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> List[str]:
        pil_images = [load_image(im) for im in images]

        texts = [
            self.processor.apply_chat_template(
                [
                    {"role": "system", "content": sp},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": pr},
                        ],
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for sp, pr in zip(system_prompts, prompts)
        ]

        inputs = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            **generate_kwargs,
        )

        # With left padding, every sequence shares the same prompt length, so the
        # newly generated tokens are exactly the columns past the input width.
        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        responses = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [r.strip() for r in responses]
