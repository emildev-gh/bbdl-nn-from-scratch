from .functions.activation import activation
from .functions.loss import loss
from .helper import initialize_weights
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
        # weights: [node][weight]; weight per input
        self.weights = initialize_weights(activation_function, num_inputs, num_nodes)
        # Inference
        self.outputs = [0.0 for _ in range(num_nodes)]
        self.net = [0.0 for _ in range(num_nodes)]
        # Learning
        self.error_signals = [0.0 for _ in range(num_nodes)]
        self.bias_gradients = [0.0 for _ in range(num_nodes)]
        # gradients: [node][gradient]; gradient per input
        self.gradients = [[0.0 for _ in range(num_inputs)] for _ in range(num_nodes)]
        # Learning: ADAM
        self.bias_m = [0.0 for _ in range(num_nodes)]
        self.bias_v = [0.0 for _ in range(num_nodes)]
        self.weight_m = [[0.0 for _ in range(num_inputs)] for _ in range(num_nodes)]
        self.weight_v = [[0.0 for _ in range(num_inputs)] for _ in range(num_nodes)]

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

    def calc_gradients(self, prev_outputs: list[float], batch_size: int = 0):
        if len(prev_outputs) != self.num_inputs:
            raise ValueError("previous outputs don't match num_inputs")
        if batch_size < 0:
            raise ValueError("batch_size cannot be negative")
        # divide each sample's contribution so summing a batch gives its mean gradient.
        scale = 1.0 if batch_size == 0 else 1.0 / batch_size
        # single-sample mode replaces the previous gradient; batch mode accumulates.
        for node in range(self.num_nodes):
            bias_gradient = self.error_signals[node] * scale
            if batch_size == 0:
                self.bias_gradients[node] = bias_gradient
            else:
                self.bias_gradients[node] += bias_gradient
            for input_node in range(self.num_inputs):
                gradient = self.error_signals[node] * prev_outputs[input_node] * scale
                if batch_size == 0:
                    self.gradients[node][input_node] = gradient
                else:
                    self.gradients[node][input_node] += gradient

    def reset_gradients(self):
        self.bias_gradients = [0.0 for _ in range(self.num_nodes)]
        self.gradients = [[0.0 for _ in range(self.num_inputs)] for _ in range(self.num_nodes)]
