import numpy as np

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

    layer.bias_m *= BETA1
    layer.bias_m += (1.0 - BETA1) * layer.bias_gradients
    layer.bias_v *= BETA2
    layer.bias_v += (1.0 - BETA2) * layer.bias_gradients**2
    corrected_m = layer.bias_m / bias_correction1
    corrected_v = layer.bias_v / bias_correction2
    layer.biases -= ALPHA * corrected_m / (np.sqrt(corrected_v) + EPSILON)

    layer.weight_m *= BETA1
    layer.weight_m += (1.0 - BETA1) * layer.gradients
    layer.weight_v *= BETA2
    layer.weight_v += (1.0 - BETA2) * layer.gradients**2
    corrected_m = layer.weight_m / bias_correction1
    corrected_v = layer.weight_v / bias_correction2
    layer.weights -= ALPHA * corrected_m / (np.sqrt(corrected_v) + EPSILON)
