from .layer import Layer
from .types import ActivationConfig, LossFunction, TrainingData

# this currently only support these configurations:
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

    def forward(self, input_values: list[float]) -> list[float]:
        self.last_inputs = input_values.copy()
        for index, layer in enumerate(self.layers):
            layer.calc_outputs(input_values if index == 0 else self.layers[index - 1].outputs)
        return self.layers[-1].outputs

    def backward(self, target_values: list[float]):
        for index, layer in reversed(list(enumerate(self.layers))):
            next_layer = self.layers[index + 1] if index + 1 < len(self.layers) else None
            layer.calc_error_signals(self.loss_function, target_values, next_layer)
        if self.last_inputs is None:
            raise ValueError("forward must be called before backward")
        for index, layer in enumerate(self.layers):
            previous_outputs = self.last_inputs if index == 0 else self.layers[index - 1].outputs
            layer.calc_gradients(previous_outputs)

    def train(self, training_data: TrainingData, epochs: int):
        if epochs < 1:
            raise ValueError("epochs must be greater than 0")
        input_data = [inputs for inputs, _ in training_data]
        target_data = [targets for _, targets in training_data]
        for _ in range(epochs):
            for instance in range(len(input_data)):
                self.forward(input_data[instance])
                self.backward(target_data[instance])
                for layer in self.layers:
                    layer.apply_gradients()

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
