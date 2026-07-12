"""A :class:`Person` reasons about images in character via a VLM backend."""

from __future__ import annotations

from typing import Optional

from .backend import ImageInput

DEFAULT_QUESTION = (
    "This image just showed up in your social media feed. Like a real person "
    "scrolling, form an honest, in-character reaction based on your own taste -- "
    "some posts you love, some you scroll past without a second thought. Give it "
    "a score from 0 to 100 for how much you personally like it, using the full "
    "range honestly instead of defaulting to the middle. Respond with ONLY a "
    "single JSON object and nothing else (no markdown, no extra text), in "
    'exactly this form: {"score": <integer 0-100>, "comment": "<your honest, '
    'in-character reaction, one or two sentences>"}.'
)


class Person:
    """A persona that forms opinions about images through a VLM backend.

    ``description`` is a free-text description of the person (personality,
    background, values, tastes...). It is turned into a system prompt so the
    backend answers *as* this person. ``backend`` is a shared VLM wrapper
    (see :class:`~persona.backend.QwenVLBackend`); many people can share one.
    """

    def __init__(self, description: str, backend, name: Optional[str] = None) -> None:
        self.name = name
        self.description = description
        self.backend = backend

    def _system_prompt(self) -> str:
        who = f"named {self.name}, " if self.name else ""
        return (
            f"You are role-playing as a person {who}described as follows:\n"
            f"{self.description}\n\n"
            "You are shown an image, as if it appeared in your social media "
            "feed. React the way this specific person actually would: let their "
            "personality, background, values, and taste -- not generic "
            "politeness -- decide whether they love it, are indifferent, or "
            "dislike it. Not everything deserves a high score; be honest and "
            "critical when the image doesn't match this person's taste. Do not "
            "mention that you are an AI or that you are role-playing."
        )

    def get_opinion(
        self,
        image: ImageInput,
        question: str = DEFAULT_QUESTION,
        **generate_kwargs,
    ) -> str:
        """Return this person's in-character opinion about ``image``."""
        return self.backend.generate(
            system_prompt=self._system_prompt(),
            image=image,
            prompt=question,
            **generate_kwargs,
        )
