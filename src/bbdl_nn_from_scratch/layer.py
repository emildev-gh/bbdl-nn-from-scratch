import random

from .constants import LEARNING_RATE
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
        ]  # [node][weight]; weight per input
        # Inference
        self.outputs = [0.0 for _ in range(num_nodes)]
        self.net = [0.0 for _ in range(num_nodes)]
        # Learning
        self.error_signals = [0.0 for _ in range(num_nodes)]
        self.gradients = [
            [0.0 for _ in range(num_inputs)] for _ in range(num_nodes)
        ]  # [node][gradient]; gradient per input

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

    def calc_error_signals(
        self, loss_function: LossFunction, target_value: list[float], next_layer: Layer | None
    ):
        if self.layer_type == "output":
            for node in range(self.num_nodes):
                derived_activation = (
                    activation(self.activation_function, self.net[node], self.net, True)
                    if loss_function == "SSE"
                    else None
                )  # this is only needed if using SSE in output layer
                self.error_signals[node] = loss(
                    loss_function,
                    self.outputs[node],
                    target_value[node],
                    derived_activation,
                )
        else:
            if next_layer is None:
                raise ValueError("next_layer is needed for hidden layers")
            for node in range(self.num_nodes):
                next_layer_weighted_error_signals = 0.0
                for next_node in range(next_layer.num_nodes):
                    next_layer_weighted_error_signals += (
                        next_layer.weights[next_node][node] * next_layer.error_signals[next_node]
                    )
                derived_activation = activation(
                    self.activation_function, self.net[node], self.net, True
                )
                self.error_signals[node] = derived_activation * next_layer_weighted_error_signals

    def calc_gradients(self, prev_outputs: list[float]):
        if len(prev_outputs) != self.num_inputs:
            raise ValueError("previous outputs don't match num_inputs")
        for node in range(self.num_nodes):
            for input_node in range(self.num_inputs):
                self.gradients[node][input_node] = (
                    self.error_signals[node] * prev_outputs[input_node]
                )

    def apply_gradients(self):
        for node in range(self.num_nodes):
            for input in range(self.num_inputs):
                self.weights[node][input] = (
                    self.weights[node][input] - LEARNING_RATE * self.gradients[node][input]
                )
