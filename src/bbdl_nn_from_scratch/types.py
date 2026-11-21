from typing import Literal

type LayerType = Literal["hidden", "output"]

type ActivationFunction = Literal["relu", "sigmoid", "softmax"]
type ActivationConfig = tuple[ActivationFunction, ActivationFunction]

type LossFunction = Literal["SSE", "CCE", "CCE-SOFTMAX"]

type InferenceData = list[list[float]]
type TrainingData = list[tuple[list[float], list[float]]]
