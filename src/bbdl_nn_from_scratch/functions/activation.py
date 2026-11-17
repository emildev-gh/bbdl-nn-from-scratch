from math import exp

from ..types import ActivationFunction


def relu(value: float) -> float:
    return max(0.0, value)


def relu_derivative(value: float) -> float:
    return 1.0 if value > 0.0 else 0.0


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def sigmoid_derivative(value: float) -> float:
    output = sigmoid(value)
    return output * (1.0 - output)


def softmax(value: float, all_values: list[float]) -> float:
    maximum = max(all_values)
    numerator = exp(value - maximum)
    denominator = sum(exp(other_value - maximum) for other_value in all_values)
    return numerator / denominator


def softmax_derivative(value: float, all_values: list[float]) -> float:
    probability = softmax(value, all_values)
    return probability * (1.0 - probability)


def activation(
    function: ActivationFunction,
    value: float,
    all_values: list[float] | None = None,
    derivative: bool = False,
) -> float:
    match function:
        case "relu":
            return relu_derivative(value) if derivative else relu(value)
        case "sigmoid":
            return sigmoid_derivative(value) if derivative else sigmoid(value)
        case "softmax":
            if all_values is None:
                raise TypeError("softmax requires all output values")
            return (
                softmax_derivative(value, all_values)
                if derivative
                else softmax(value, all_values)
            )
        case _:
            raise ValueError(f"Unsupported activation function: {function}")
