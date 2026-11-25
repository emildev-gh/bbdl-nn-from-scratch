import random
from math import sqrt

from .constants import BATCH_SIZE
from .types import (
    ActivationConfig,
    ActivationFunction,
    InferenceData,
    LossFunction,
    TrainingData,
    TrainingDataBatched,
)


def initialize_weights(
    activation_function: ActivationFunction,
    num_inputs: int,
    num_nodes: int,
) -> list[list[float]]:
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
    return weights


def build_batches(data: TrainingData) -> TrainingDataBatched:
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    if BATCH_SIZE == 0:
        return [shuffled_data]
    batches: TrainingDataBatched = []
    for start in range(0, len(shuffled_data), BATCH_SIZE):
        batches.append(shuffled_data[start : start + BATCH_SIZE])
    return batches


def extract_input_data(training_data: TrainingData) -> InferenceData:
    return [inputs for inputs, _ in training_data]


def extract_target_data(training_data: TrainingData) -> InferenceData:
    return [targets for _, targets in training_data]


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
