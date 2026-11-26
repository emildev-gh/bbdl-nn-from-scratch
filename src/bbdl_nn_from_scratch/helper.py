from __future__ import annotations

import random
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np

from .constants import BATCH_SIZE
from .types import (
    ActivationConfig,
    ActivationFunction,
    FloatArray,
    LossFunction,
    Optimizer,
    TrainingData,
    TrainingDataBatched,
)

if TYPE_CHECKING:
    from .neuralnetwork import NeuralNetwork


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


def calc_accuracy(network: NeuralNetwork, validation_data: TrainingData) -> float:
    inputs, targets = validation_data
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("validation inputs and targets must be two-dimensional")
    if len(inputs) == 0:
        raise ValueError("validation data must not be empty")
    if len(inputs) != len(targets):
        raise ValueError("validation inputs and targets must have the same number of samples")

    target_labels = np.argmax(targets, axis=1)
    correct = 0
    for input_data, target_label in zip(inputs, target_labels, strict=True):
        prediction = network.run(input_data.tolist())
        if len(prediction) != targets.shape[1]:
            raise ValueError("network output size must match the number of target classes")
        correct += int(np.argmax(prediction) == target_label)
    return float(correct / len(inputs) * 100.0)


def validate_training_run(
    training_data: TrainingData,
    epochs: int | None,
    optimizer: Optimizer,
    desired_accuracy: float | None,
    validation_data: TrainingData | None,
) -> None:
    if epochs is None and desired_accuracy is None:
        raise ValueError("epochs or desired_accuracy must be provided")
    if epochs is not None and (isinstance(epochs, bool) or not isinstance(epochs, int)):
        raise TypeError("epochs must be an integer or None")
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be greater than 0")
    if optimizer not in ("ADAM", "SGD"):
        raise ValueError(f"unsupported optimizer: {optimizer}")
    if desired_accuracy is not None:
        if not 0.0 <= desired_accuracy <= 100.0:
            raise ValueError("desired_accuracy must be between 0 and 100")
        if validation_data is None:
            raise ValueError("validation_data is required with desired_accuracy")
        if len(training_data[0]) == 0 or len(validation_data[0]) == 0:
            raise ValueError("training and validation data must not be empty")


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
