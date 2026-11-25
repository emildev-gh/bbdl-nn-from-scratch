import numpy as np

from .types import FloatArray

# AI generated

_ASCII_GRAYSCALE = " .:-=+*#%@"
_IMAGE_SHAPE = (28 * 28,)
_TARGET_SHAPE = (10,)


def display_image(sample: tuple[FloatArray, FloatArray]) -> None:
    image, target = sample
    if image.shape != _IMAGE_SHAPE:
        raise ValueError(f"image must have shape {_IMAGE_SHAPE}, got {image.shape}")
    if target.shape != _TARGET_SHAPE:
        raise ValueError(f"target must have shape {_TARGET_SHAPE}, got {target.shape}")

    max_intensity = len(_ASCII_GRAYSCALE) - 1
    for row in image.reshape(28, 28):
        print(
            "".join(
                _ASCII_GRAYSCALE[int(np.clip(pixel, 0.0, 1.0) * max_intensity)] for pixel in row
            )
        )

    print(f"Target label: {int(np.argmax(target))}")
