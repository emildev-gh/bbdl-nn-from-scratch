from typing import Literal

type LayerType = Literal["hidden", "output"]

type ActivationFunction = Literal["relu", "sigmoid", "softmax"]
type ActivationConfig = tuple[ActivationFunction, ActivationFunction]

type LossFunction = Literal["SSE", "CCE-SOFTMAX"]
type Optimizer = Literal["adam", "sgd"]

type InferenceData = list[list[float]]
type TrainingData = list[tuple[list[float], list[float]]]
type TrainingDataBatched = list[TrainingData]
