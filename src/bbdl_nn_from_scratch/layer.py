import numpy as np

from .functions.activation import activation_vectorized
from .functions.loss import loss
from .helper import initialize_weights
from .types import ActivationFunction, FloatArray, LayerType, LossFunction


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
        self.num_inputs: int = num_inputs
        self.num_nodes: int = num_nodes
        self.biases: FloatArray = np.zeros(num_nodes)
        # weights: [node][weight]; weight per input
        self.weights: FloatArray = initialize_weights(activation_function, num_inputs, num_nodes)
        # Inference
        self.outputs: FloatArray = np.zeros((1, num_nodes))
        self.net: FloatArray = np.zeros((1, num_nodes))
        # Learning
        self.error_signals: FloatArray = np.zeros((1, num_nodes))
        self.bias_gradients: FloatArray = np.zeros(num_nodes)
        # gradients: [node][gradient]; gradient per input
        self.gradients: FloatArray = np.zeros((num_nodes, num_inputs))
        # Learning: ADAM
        self.bias_m: FloatArray = np.zeros(num_nodes)
        self.bias_v: FloatArray = np.zeros(num_nodes)
        self.weight_m: FloatArray = np.zeros((num_nodes, num_inputs))
        self.weight_v: FloatArray = np.zeros((num_nodes, num_inputs))

    def calc_outputs(self, inputs: FloatArray) -> FloatArray:
        if inputs.ndim != 2 or inputs.shape[1] != self.num_inputs:
            raise ValueError("inputs must have shape (batch_size, num_inputs)")

        self.net = inputs @ self.weights.T + self.biases
        self.outputs = activation_vectorized(self.activation_function, self.net)
        return self.outputs

    def calc_error_signals(
        self, loss_function: LossFunction, target_values: FloatArray, next_layer: Layer | None
    ) -> None:
        if self.layer_type == "OUTPUT":
            if loss_function == "CCE-SOFTMAX":
                self.error_signals = self.outputs - target_values
            else:
                activation_derivatives = activation_vectorized(
                    self.activation_function, self.net, derivative=True
                )
                self.error_signals = np.asarray(
                    loss(loss_function, self.outputs, target_values, activation_derivatives),
                    dtype=float,
                )
        else:
            if next_layer is None:
                raise ValueError("next_layer is needed for hidden layers")
            activation_derivatives = activation_vectorized(
                self.activation_function, self.net, derivative=True
            )
            self.error_signals = (
                next_layer.error_signals @ next_layer.weights
            ) * activation_derivatives

    def calc_gradients(self, prev_outputs: FloatArray) -> None:
        if prev_outputs.ndim != 2 or prev_outputs.shape[1] != self.num_inputs:
            raise ValueError("previous outputs don't match num_inputs")
        if len(prev_outputs) != len(self.error_signals):
            raise ValueError("previous outputs and error signals must have the same batch size")
        self.gradients = self.error_signals.T @ prev_outputs / len(prev_outputs)
        self.bias_gradients = np.mean(self.error_signals, axis=0)

    def reset_gradients(self) -> None:
        self.bias_gradients = np.zeros(self.num_nodes)
        self.gradients = np.zeros((self.num_nodes, self.num_inputs))
