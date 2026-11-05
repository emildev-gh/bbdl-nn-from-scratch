from math import exp

from .types import LayerType


def relu(value: float) -> float:
    return max(0.0, value)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def activation(type: LayerType, value: float) -> float:
    match type:
        case "hidden":
            return relu(value)
        case "output":
            return sigmoid(value)
        case _:
            raise ValueError("Unknown layer type")
