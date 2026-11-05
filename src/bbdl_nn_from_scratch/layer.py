import random

from .activation import activation
from .types import LayerType


class Layer:
    def __init__(
        self,
        layer_type: LayerType,
        num_inputs: int,
        num_nodes: int,
    ):
        self.layer_type: LayerType = layer_type
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

        for node in range(self.num_nodes):
            weightedInputs = 0
            for input_index, input_value in enumerate(inputs):
                weightedInputs += input_value * self.weights[node][input_index]
            self.net[node] = weightedInputs + self.biases[node]
            self.outputs[node] = activation(self.layer_type, self.net[node])
        return self.outputs
