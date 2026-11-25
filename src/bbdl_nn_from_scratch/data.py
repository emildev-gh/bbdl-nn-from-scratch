from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .constants import RANDOM_SEED
from .types import FloatArray, TrainingData

# AI generated


def load_data(filename: str) -> TrainingData:
    print(f"loading file '{filename}'...")
    data_file = Path(__file__).resolve().parents[2] / "data" / filename
    frame = pd.read_parquet(data_file, columns=["image", "label"])
    inputs: list[FloatArray] = []
    labels: list[int] = []

    for image_record, label in frame.itertuples(index=False, name=None):
        with Image.open(BytesIO(image_record["bytes"])) as image:
            inputs.append(np.asarray(image.convert("L"), dtype=float).reshape(-1) / 255.0)
        labels.append(label)

    input_data = np.stack(inputs) if inputs else np.empty((0, 0), dtype=float)
    target_data = np.eye(10, dtype=float)[labels]
    return input_data, target_data


def load_data_split(
    filename: str, training_size: int = 50_000
) -> tuple[TrainingData, TrainingData]:
    if isinstance(training_size, bool) or not isinstance(training_size, int):
        raise TypeError("training_size must be an integer")
    if training_size < 1:
        raise ValueError("training_size must be greater than 0")

    input_data, target_data = load_data(filename)
    if training_size >= len(input_data):
        raise ValueError(f"training_size must be smaller than the dataset size ({len(input_data)})")

    indices = np.random.default_rng(RANDOM_SEED).permutation(len(input_data))
    training_indices = indices[:training_size]
    validation_indices = indices[training_size:]
    return (
        (input_data[training_indices], target_data[training_indices]),
        (input_data[validation_indices], target_data[validation_indices]),
    )
