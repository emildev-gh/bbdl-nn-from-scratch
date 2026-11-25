import random
from math import sqrt

import numpy as np

from .constants import BATCH_SIZE
from .types import (
    ActivationConfig,
    ActivationFunction,
    FloatArray,
    LossFunction,
    TrainingData,
    TrainingDataBatched,
)


def initialize_weights(
    activation_function: ActivationFunction,
    num_inputs: int,
    num_nodes: int,
) -> FloatArray:
    if num_inputs < 1:
        raise ValueError("num_inputs must be greater than 0")
    if num_nodes < 1:
        raise ValueError("num_nodes must be greater than 0")

    if activation_function == "RELU":
        standard_deviation = sqrt(2.0 / num_inputs)
        weights = [
            [random.gauss(0.0, standard_deviation) for _ in range(num_inputs)]
            for _ in range(num_nodes)
        ]
    else:
        limit = 1.0 / sqrt(num_inputs)
        weights = [
            [random.uniform(-limit, limit) for _ in range(num_inputs)] for _ in range(num_nodes)
        ]

    for node_weights in weights:
        for index, weight in enumerate(node_weights):
            if weight == 0.0:
                node_weights[index] = 0.01
    return np.asarray(weights, dtype=float)


def build_batches(data: TrainingData) -> TrainingDataBatched:
    inputs, targets = data
    indices = list(range(len(inputs)))
    random.shuffle(indices)
    shuffled_indices = np.asarray(indices, dtype=int)
    if BATCH_SIZE == 0:
        return [(inputs[shuffled_indices], targets[shuffled_indices])]
    batches: TrainingDataBatched = []
    for start in range(0, len(inputs), BATCH_SIZE):
        batch_indices = shuffled_indices[start : start + BATCH_SIZE]
        batches.append((inputs[batch_indices], targets[batch_indices]))
    return batches


def validate_network_configuration(
    input_size: int,
    hidden_size: list[int],
    output_size: int,
    activation_config: ActivationConfig,
    loss_function: LossFunction,
) -> None:
    if input_size < 1:
        raise ValueError("input_size must be greater than 0")
    if output_size < 1:
        raise ValueError("output_size must be greater than 0")
    if any(num_nodes < 1 for num_nodes in hidden_size):
        raise ValueError("hidden layer sizes must be greater than 0")
    if hidden_size and activation_config[0] not in ("RELU", "SIGMOID"):
        raise ValueError("hidden activation must be 'RELU' or 'SIGMOID'")
    if loss_function == "SSE":
        if activation_config[1] not in ("RELU", "SIGMOID"):
            raise ValueError("SSE requires a 'RELU' or 'SIGMOID' output activation")
    elif loss_function == "CCE-SOFTMAX":
        if activation_config[1] != "SOFTMAX":
            raise ValueError("CCE-SOFTMAX requires a 'SOFTMAX' output activation")
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")
