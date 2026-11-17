
import numpy as np
from PIL import Image
from pathlib import Path


def load_image(image_path):
    """Load an image and convert it to RGB format."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    return np.array(image)


def save_image(image_array, output_path):
    """Save a NumPy image array as an image file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_array = np.clip(
        image_array, 0, 255
    ).astype(np.uint8)

    image = Image.fromarray(image_array)
    image.save(output_path)

    print(f"Image saved successfully: {output_path}")
