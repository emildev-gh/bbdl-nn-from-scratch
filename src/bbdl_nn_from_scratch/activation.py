from math import exp

from .types import ActivationFunction


def relu(value: float) -> float:
    return max(0.0, value)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def softmax(value: float, all_values: list[float]) -> float:
    maximum = max(all_values)
    numerator = exp(value - maximum)
    denominator = sum(exp(other_value - maximum) for other_value in all_values)
    return numerator / denominator


def activation(
    function: ActivationFunction,
    value: float,
    all_values: list[float] | None = None,
) -> float:
    match function:
        case "relu":
            return relu(value)
        case "sigmoid":
            return sigmoid(value)
        case "softmax":
            if all_values is None:
                raise TypeError("softmax requires all output values")
            return softmax(value, all_values)
        case _:
            raise ValueError(f"Unsupported activation function: {function}")
