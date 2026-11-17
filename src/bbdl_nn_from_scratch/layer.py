import random

from .functions.activation import activation
from .functions.loss import loss
from .types import ActivationFunction, LayerType, LossFunction


class Layer:
    def __init__(
        self,
        layer_type: LayerType,
        activation_function: ActivationFunction,
        num_inputs: int,
        num_nodes: int,
    ):
        # Layer
        self.layer_type: LayerType = layer_type
        self.activation_function: ActivationFunction = activation_function
        self.num_inputs = num_inputs
        self.num_nodes = num_nodes
        self.biases = [0.0 for _ in range(num_nodes)]
        self.weights = [
            [random.uniform(-1, 1) for _ in range(num_inputs)] for _ in range(num_nodes)
        ]  # [node][weight]
        # Inference
        self.outputs = [0.0 for _ in range(num_nodes)]
        self.net = [0.0 for _ in range(num_nodes)]
        # Learning
        self.error_signals = [0.0 for _ in range(num_nodes)]

    def calc_outputs(self, inputs: list[float]) -> list[float]:
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
            self.outputs[node] = activation(self.activation_function, self.net[node], self.net)
        return self.outputs

    def calc_error_signals(self, loss_function: LossFunction, target_value: list[float]):
        if self.layer_type == "output":
            for node in range(self.num_nodes):
                derived_activation = (
                    activation(self.activation_function, self.net[node], self.net, True)
                    if loss_function == "SSE"
                    else None
                )
                self.error_signals[node] = loss(
                    loss_function,
                    self.outputs[node],
                    target_value[node],
                    derived_activation,
                )
