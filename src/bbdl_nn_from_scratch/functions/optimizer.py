from math import sqrt

from ..constants import ALPHA, BETA1, BETA2, EPSILON, LEARNING_RATE
from ..layer import Layer


def apply_gradients_sgd(layer: Layer) -> None:
    for node in range(layer.num_nodes):
        layer.biases[node] -= LEARNING_RATE * layer.bias_gradients[node]
        for input_node in range(layer.num_inputs):
            layer.weights[node][input_node] -= LEARNING_RATE * layer.gradients[node][input_node]


def apply_gradients_adam(layer: Layer, step: int) -> None:
    if step < 1:
        raise ValueError("step must be greater than 0")

    bias_correction1 = 1.0 - BETA1**step
    bias_correction2 = 1.0 - BETA2**step

    for node in range(layer.num_nodes):
        gradient = layer.bias_gradients[node]
        layer.bias_m[node] = BETA1 * layer.bias_m[node] + (1.0 - BETA1) * gradient
        layer.bias_v[node] = BETA2 * layer.bias_v[node] + (1.0 - BETA2) * gradient**2
        corrected_m = layer.bias_m[node] / bias_correction1
        corrected_v = layer.bias_v[node] / bias_correction2
        layer.biases[node] -= ALPHA * corrected_m / (sqrt(corrected_v) + EPSILON)

        for input_node in range(layer.num_inputs):
            gradient = layer.gradients[node][input_node]
            layer.weight_m[node][input_node] = (
                BETA1 * layer.weight_m[node][input_node] + (1.0 - BETA1) * gradient
            )
            layer.weight_v[node][input_node] = (
                BETA2 * layer.weight_v[node][input_node] + (1.0 - BETA2) * gradient**2
            )
            corrected_m = layer.weight_m[node][input_node] / bias_correction1
            corrected_v = layer.weight_v[node][input_node] / bias_correction2
            layer.weights[node][input_node] -= ALPHA * corrected_m / (sqrt(corrected_v) + EPSILON)
