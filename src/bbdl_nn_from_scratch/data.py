from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from .types import TrainingData


def load_data(filename: str) -> TrainingData:
    print(f"loading file '{filename}'...")
    data_file = Path(__file__).resolve().parents[2] / "data" / filename
    frame = pd.read_parquet(data_file, columns=["image", "label"])
    data: TrainingData = []

    for image_record, label in frame.itertuples(index=False, name=None):
        with Image.open(BytesIO(image_record["bytes"])) as image:
            pixels = image.convert("L").getdata()
            inputs = []
            for pixel in pixels:
                if not isinstance(pixel, int):
                    raise TypeError("expected grayscale pixel values")
                inputs.append(pixel / 255.0)

        targets = [0.0] * 10
        targets[label] = 1.0
        data.append((inputs, targets))

    return data
