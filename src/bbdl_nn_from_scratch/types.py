from typing import Literal

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

type LayerType = Literal["HIDDEN", "OUTPUT"]

type ActivationFunction = Literal["RELU", "SIGMOID", "SOFTMAX"]
type ActivationConfig = tuple[ActivationFunction, ActivationFunction]

type LossFunction = Literal["SSE", "CCE-SOFTMAX"]
type Optimizer = Literal["ADAM", "SGD"]

type InferenceData = list[list[float]]
type TrainingData = tuple[FloatArray, FloatArray]
type TrainingDataBatched = list[TrainingData]
