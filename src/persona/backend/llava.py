"""LLaVA backend used for persona reasoning over images.

Unlike Qwen2-VL, the LLaVA-1.5 chat template has no system slot and silently
drops any ``system`` message, so this backend folds the persona prompt into the
user turn to make sure it actually conditions the response.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .base import ImageInput, VLMBackend, load_image


class LlavaBackend(VLMBackend):
    """LLaVA backend (defaults to ``llava-hf/llava-1.5-7b-hf``).

    Parameters
    ----------
    model_name:
        Hugging Face repo id of a ``llava-hf`` checkpoint.
    device_map:
        Passed to ``from_pretrained``; ``"auto"`` lets Accelerate place the
        weights (a 7B model fits comfortably on a single GPU).
    dtype:
        Compute dtype for the weights. ``bfloat16`` is a good default on
        Ampere/Hopper GPUs.
    max_new_tokens:
        Default generation length for :meth:`generate`.
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def generate(
        self,
        system_prompt: str,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> str:
        pil_image = load_image(image)

        # LLaVA-1.5 has no system role, so prepend the persona to the user turn.
        user_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            **generate_kwargs,
        )

        # Drop the prompt tokens so we only decode the newly generated answer.
        new_tokens = [
            output[len(prompt_ids):]
            for prompt_ids, output in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()
