"""Demo: give an image to a few different people and print their opinions.

Run from the repo root:

    python examples/opinions.py path/to/image.jpg [--model-name Qwen/Qwen2-VL-7B-Instruct]

The image argument is optional; without it a small test image is generated.
One Qwen2-VL backend is loaded once and shared across all the people.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from persona import Person, QwenVLBackend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Give an image to a few different people and print their opinions."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to an image file. If omitted, a small demo image is generated.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Qwen2-VL model name to load (default: %(default)s).",
    )
    return parser.parse_args()


def _demo_image() -> Image.Image:
    """A trivially generated image so the demo runs with no arguments."""
    from PIL import ImageDraw

    img = Image.new("RGB", (512, 512), (135, 206, 235))  # sky blue
    draw = ImageDraw.Draw(img)
    draw.ellipse((380, 40, 470, 130), fill=(255, 221, 0))       # sun
    draw.rectangle((0, 380, 512, 512), fill=(34, 139, 34))       # grass
    draw.polygon([(120, 380), (220, 200), (320, 380)], fill=(90, 90, 90))  # mountain
    return img


def main() -> None:
    args = _parse_args()
    image = args.image if args.image is not None else _demo_image()

    # Load the VLM once; share it across everyone.
    backend = QwenVLBackend(model_name=args.model_name)

    people = [
        Person(
            name="Maya",
            description=(
                "a cheerful 8-year-old who loves nature, animals, and drawing. "
                "Easily excited and speaks simply."
            ),
            backend=backend,
        ),
        Person(
            name="Viktor",
            description=(
                "a stern, world-weary art critic with 40 years of experience. "
                "Hard to impress, precise, and a little pretentious."
            ),
            backend=backend,
        ),
        Person(
            name="Dr. Chen",
            description=(
                "a pragmatic environmental scientist who analyzes everything "
                "through the lens of ecology and data."
            ),
            backend=backend,
        ),
    ]

    for person in people:
        opinion = person.get_opinion(image)
        print(f"\n=== {person.name} ===\n{opinion}")


if __name__ == "__main__":
    main()
