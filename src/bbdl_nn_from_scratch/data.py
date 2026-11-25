from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .types import FloatArray, TrainingData


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
