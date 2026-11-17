import random

from .functions.activation import activation
from .types import ActivationFunction, LayerType


class Layer:
    def __init__(
        self,
        layer_type: LayerType,
        activation_function: ActivationFunction,
        num_inputs: int,
        num_nodes: int,
    ):
        self.layer_type: LayerType = layer_type
        self.activation_function: ActivationFunction = activation_function
        self.num_inputs = num_inputs
        self.num_nodes = num_nodes
        self.biases = [0.0 for _ in range(num_nodes)]
        self.weights = [
            [random.uniform(-1, 1) for _ in range(num_inputs)] for _ in range(num_nodes)
        ]  # [node][weight]
        self.outputs = [0.0 for _ in range(num_nodes)]
        self.net = [0.0 for _ in range(num_nodes)]

    def calculate_outputs(self, inputs: list[float]) -> list[float]:
        if len(inputs) != self.num_inputs:
            raise ValueError("num_inputs doesn't match len(inputs)")

        # calculate nets
        for node in range(self.num_nodes):
            weighted_inputs = 0.0
            for input_index, input_value in enumerate(inputs):
                weighted_inputs += input_value * self.weights[node][input_index]
            self.net[node] = weighted_inputs + self.biases[node]

        # calculate outputs
        for node in range(self.num_nodes):
            self.outputs[node] = activation(
                self.activation_function, self.net[node], self.net
            )
        return self.outputs
