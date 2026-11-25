import random
from pathlib import Path

import numpy as np

from .constants import BATCH_SIZE, RANDOM_SEED
from .functions.optimizer import apply_gradients_adam, apply_gradients_sgd
from .helper import (
    build_batches,
    extract_input_data,
    extract_target_data,
    validate_network_configuration,
)
from .layer import Layer
from .persistence import load_network, save_network
from .types import ActivationConfig, FloatArray, LossFunction, Optimizer, TrainingData

# this currently only supports these configurations:
# - hidden: relu or sigmoid
# - output: sigmoid/relu + SSE
# - output: softmax + CCE-SOFTMAX


class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_size: list[int],
        output_size: int,
        activation_config: ActivationConfig,
        loss_function: LossFunction,
    ) -> None:
        validate_network_configuration(
            input_size,
            hidden_size,
            output_size,
            activation_config,
            loss_function,
        )

        random.seed(RANDOM_SEED)
        self.layers: list[Layer] = []
        for num_nodes in hidden_size:
            self.layers.append(
                Layer(
                    "HIDDEN",
                    activation_config[0],
                    input_size if not self.layers else self.layers[-1].num_nodes,
                    num_nodes,
                )
            )
        previous_size = self.layers[-1].num_nodes if self.layers else input_size
        self.layers.append(Layer("OUTPUT", activation_config[1], previous_size, output_size))
        self.loss_function: LossFunction = loss_function
        self.adam_step: int = 0
        # Caching last used inputs for a forward pass
        self._last_inputs_cache: list[float] | None = None
        self._batch_inputs_cache: FloatArray | None = None

    def _forward_vectorized(self, input_values: FloatArray) -> FloatArray:
        if input_values.ndim != 2:
            raise ValueError("input_values must be a two-dimensional batch")
        self._batch_inputs_cache = input_values.copy()
        self._last_inputs_cache = input_values[-1].tolist()
        for index, layer in enumerate(self.layers):
            layer.calc_outputs(input_values if index == 0 else self.layers[index - 1].outputs)
        return self.layers[-1].outputs

    def _forward(self, input_values: list[float]) -> list[float]:
        outputs = self._forward_vectorized(np.asarray(input_values, dtype=float)[None, :])
        return outputs[0].tolist()

    def _backward_vectorized(self, target_values: FloatArray) -> None:
        for index, layer in reversed(list(enumerate(self.layers))):
            next_layer = self.layers[index + 1] if index + 1 < len(self.layers) else None
            layer.calc_error_signals(self.loss_function, target_values, next_layer)
        if self._batch_inputs_cache is None:
            raise ValueError("forward must be called before backward")
        for index, layer in enumerate(self.layers):
            previous_outputs = (
                self._batch_inputs_cache if index == 0 else self.layers[index - 1].outputs
            )
            layer.calc_gradients(previous_outputs)

    def _apply_optimizer_step(self, optimizer: Optimizer) -> None:
        if optimizer == "ADAM":
            self.adam_step += 1
            for layer in self.layers:
                apply_gradients_adam(layer, self.adam_step)
        elif optimizer == "SGD":
            for layer in self.layers:
                apply_gradients_sgd(layer)
        else:
            raise ValueError(f"unsupported optimizer: {optimizer}")

    def train(
        self,
        training_data: TrainingData,
        epochs: int,
        model_name: str | None = None,
        optimizer: Optimizer = "ADAM",
    ):
        if epochs < 1:
            raise ValueError("epochs must be greater than 0")
        if optimizer not in ("ADAM", "SGD"):
            raise ValueError(f"unsupported optimizer: {optimizer}")
        for _ in range(epochs):
            training_data_batched = build_batches(training_data)
            for batch in training_data_batched:
                for layer in self.layers:
                    layer.reset_gradients()
                input_data = extract_input_data(batch)
                target_data = extract_target_data(batch)
                if BATCH_SIZE == 0:
                    for instance in range(len(batch)):
                        self._forward_vectorized(input_data[instance : instance + 1])
                        self._backward_vectorized(target_data[instance : instance + 1])
                        self._apply_optimizer_step(optimizer)
                else:
                    self._forward_vectorized(input_data)
                    self._backward_vectorized(target_data)
                    self._apply_optimizer_step(optimizer)
        if model_name is not None:
            self.save(model_name, epochs)

    def run(self, inputs: list[float]) -> list[float]:
        return self._forward(inputs)

    # --- Miscellaneous ----------------

    def save(self, file_name: str, epochs_trained_on: int | None = None) -> None:
        save_network(self, f"./models/{file_name}", epochs_trained_on)

    @classmethod
    def load(cls, path: str | Path) -> NeuralNetwork:
        network = load_network(path)
        if not isinstance(network, cls):
            raise TypeError("loaded network type does not match the requested class")
        return network

    def printNetwork(self) -> None:
        print(f"Input ({self.layers[0].num_inputs}) [Last used values: {self._last_inputs_cache}]")
        for index, layer in enumerate(self.layers, start=1):
            print("    |")
            print(
                f"Layer {index} ({layer.layer_type}, {layer.activation_function}, "
                f"{layer.num_nodes} nodes)"
            )
            for node in range(layer.num_nodes):
                weights = ", ".join(f"{weight:.6f}" for weight in layer.weights[node])
                print(
                    f"  L{index}N{node}: weights=[{weights}], "
                    f"output={layer.outputs[-1][node]:.6f}, "
                    f"error_signal={layer.error_signals[-1][node]:.6f}"
                )
        outputs = self.layers[-1].outputs[-1].tolist()
        print(f"Output ({len(outputs)}): {outputs}")
