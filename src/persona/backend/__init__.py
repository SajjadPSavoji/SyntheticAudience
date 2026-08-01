"""VLM backends.

Subclass :class:`VLMBackend` to add a new model family, then export it here.
Each backend is loaded once and shared across many ``Person`` instances.
"""

from .base import ImageInput, ImagesInput, VLMBackend, load_image, load_images
from .llava import LlavaBackend
from .qwen import QwenVLBackend

__all__ = [
    "VLMBackend",
    "QwenVLBackend",
    "LlavaBackend",
    "ImageInput",
    "ImagesInput",
    "load_image",
    "load_images",
]
