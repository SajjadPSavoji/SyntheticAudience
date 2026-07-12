"""VLM backends.

Subclass :class:`VLMBackend` to add a new model family, then export it here.
Each backend is loaded once and shared across many ``Person`` instances.
"""

from .base import ImageInput, VLMBackend, load_image
from .llava import LlavaBackend
from .qwen import QwenVLBackend

__all__ = ["VLMBackend", "QwenVLBackend", "LlavaBackend", "ImageInput", "load_image"]
