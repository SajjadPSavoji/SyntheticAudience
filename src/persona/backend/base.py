"""Base interface and shared helpers for VLM backends.

A backend wraps a vision-language model and turns a persona (system prompt),
an image, and a user prompt into a text response. Subclass :class:`VLMBackend`
to add a new model family (Qwen, LLaVA, InternVL, ...). A backend is loaded
once and shared across many :class:`~persona.person.Person` instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Union

from PIL import Image

ImageInput = Union[str, Path, Image.Image]


def load_image(image: ImageInput) -> Image.Image:
    """Return an RGB :class:`PIL.Image.Image` from a path or an image object."""
    if isinstance(image, Image.Image):
        pil = image
    else:
        pil = Image.open(image)
    return pil.convert("RGB")


class VLMBackend(ABC):
    """Abstract base class for a vision-language-model backend.

    Concrete backends load a model once (typically in ``__init__``) and
    implement :meth:`generate`.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> str:
        """Run the VLM on one image and return the decoded text response.

        ``system_prompt`` sets the persona, ``prompt`` is the user turn, and
        ``image`` is shown to the model. Extra ``generate_kwargs`` are
        forwarded to the underlying generation call.
        """
        raise NotImplementedError

    def generate_batch(
        self,
        system_prompts: Sequence[str],
        images: Sequence[ImageInput],
        prompts: Sequence[str],
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> List[str]:
        """Run the VLM on a batch of (system_prompt, image, prompt) triples.

        The default implementation just loops :meth:`generate`, so every backend
        works out of the box. Backends whose runtime supports padded batched
        decoding (e.g. Qwen2-VL) should override this to run one fused
        ``model.generate`` over the whole batch, which is where the real speedup
        comes from. Returns one decoded string per input, in order.
        """
        return [
            self.generate(sp, im, pr, max_new_tokens=max_new_tokens, **generate_kwargs)
            for sp, im, pr in zip(system_prompts, images, prompts)
        ]
