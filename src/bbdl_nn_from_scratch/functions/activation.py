from math import exp

import numpy as np

from ..types import ActivationFunction, FloatArray


def activation_vectorized(
    function: ActivationFunction, values: FloatArray, derivative: bool = False
) -> FloatArray:
    match function:
        case "RELU":
            return (values > 0.0).astype(float) if derivative else np.maximum(values, 0.0)
        case "SIGMOID":
            outputs = np.empty_like(values)
            positive = values >= 0.0
            outputs[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
            exp_values = np.exp(values[~positive])
            outputs[~positive] = exp_values / (1.0 + exp_values)
            return outputs * (1.0 - outputs) if derivative else outputs
        case "SOFTMAX":
            shifted_values = values - np.max(values, axis=1, keepdims=True)
            exp_values = np.exp(shifted_values)
            outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            return outputs * (1.0 - outputs) if derivative else outputs
        case _:
            raise ValueError(f"Unsupported activation function: {function}")


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
        case "RELU":
            return relu_derivative(value) if derivative else relu(value)
        case "SIGMOID":
            return sigmoid_derivative(value) if derivative else sigmoid(value)
        case "SOFTMAX":
            if all_values is None:
                raise TypeError("softmax requires all output values")
            return (
                softmax_derivative(value, all_values) if derivative else softmax(value, all_values)
            )
        case _:
            raise ValueError(f"Unsupported activation function: {function}")
