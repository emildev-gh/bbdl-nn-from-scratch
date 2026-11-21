from pathlib import Path

from .helper import extract_input_data, extract_target_data
from .layer import Layer
from .persistence import load_network, save_network
from .types import ActivationConfig, LossFunction, TrainingData

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
        self.layers: list[Layer] = []
        for num_nodes in hidden_size:
            self.layers.append(
                Layer(
                    "hidden",
                    activation_config[0],
                    input_size if not self.layers else self.layers[-1].num_nodes,
                    num_nodes,
                )
            )
        self.layers.append(
            Layer("output", activation_config[1], self.layers[-1].num_nodes, output_size)
        )
        self.loss_function: LossFunction = loss_function
        self.last_inputs: list[float] | None = None  # Caching last used inputs for a forward pass

    def _forward(self, input_values: list[float]) -> list[float]:
        self.last_inputs = input_values.copy()
        for index, layer in enumerate(self.layers):
            layer.calc_outputs(input_values if index == 0 else self.layers[index - 1].outputs)
        return self.layers[-1].outputs

    def _backward(self, target_values: list[float]):
        for index, layer in reversed(list(enumerate(self.layers))):
            next_layer = self.layers[index + 1] if index + 1 < len(self.layers) else None
            layer.calc_error_signals(self.loss_function, target_values, next_layer)
        if self.last_inputs is None:
            raise ValueError("forward must be called before backward")
        for index, layer in enumerate(self.layers):
            previous_outputs = self.last_inputs if index == 0 else self.layers[index - 1].outputs
            layer.calc_gradients(previous_outputs)

    def train(self, training_data: TrainingData, epochs: int, model_name: str | None = None):
        if epochs < 1:
            raise ValueError("epochs must be greater than 0")
        input_data = extract_input_data(training_data)
        target_data = extract_target_data(training_data)
        for _ in range(epochs):
            for instance in range(len(input_data)):
                self._forward(input_data[instance])
                self._backward(target_data[instance])
                for layer in self.layers:
                    layer.apply_gradients()
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

    def printNetwork(self):
        print(f"Input ({self.layers[0].num_inputs}) [Last used input values: {self.last_inputs}]")
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
                    f"output={layer.outputs[node]:.6f}, "
                    f"error_signal={layer.error_signals[node]:.6f}"
                )
        print(f"Output ({len(self.layers[-1].outputs)}): {self.layers[-1].outputs}")
