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

# One turn's visual input: a single image, or several to show side by side (e.g.
# a pairwise A-vs-B comparison). Backends whose chat template can interleave
# multiple images accept the sequence form; the rest see only single images.
ImagesInput = Union[ImageInput, Sequence[ImageInput]]


def load_image(image: ImageInput) -> Image.Image:
    """Return an RGB :class:`PIL.Image.Image` from a path or an image object."""
    if isinstance(image, Image.Image):
        pil = image
    else:
        pil = Image.open(image)
    return pil.convert("RGB")


def load_images(images: ImagesInput) -> List[Image.Image]:
    """Normalize one turn's visual input to a list of RGB images.

    A bare image (path or PIL object) becomes a one-element list, so callers and
    backends can treat the single- and multi-image cases uniformly.
    """
    if isinstance(images, (str, Path, Image.Image)):
        return [load_image(images)]
    return [load_image(image) for image in images]


class VLMBackend(ABC):
    """Abstract base class for a vision-language-model backend.

    Concrete backends load a model once (typically in ``__init__``) and
    implement :meth:`generate`.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        image: ImagesInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> str:
        """Run the VLM on one image and return the decoded text response.

        ``system_prompt`` sets the persona, ``prompt`` is the user turn, and
        ``image`` is shown to the model — either a single image or a sequence of
        them, for backends that can interleave several in one turn. Extra
        ``generate_kwargs`` are forwarded to the underlying generation call.
        """
        raise NotImplementedError

    def generate_batch(
        self,
        system_prompts: Sequence[str],
        images: Sequence[ImagesInput],
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
