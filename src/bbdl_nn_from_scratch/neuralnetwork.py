from .layer import Layer


class NeuralNetwork:
    def __init__(
        self, input_size: int, hidden_size: list[int], output_size: int
    ) -> None:
        self.layers = []
        for num_nodes in hidden_size:
            self.layers.append(
                Layer(
                    "hidden",
                    input_size if not self.layers else self.layers[-1].num_nodes,
                    num_nodes,
                )
            )
        self.layers.append(Layer("output", self.layers[-1].num_nodes, output_size))

    def forward(self, inputs: list[float]) -> list[float]:
        for index, layer in enumerate(self.layers):
            layer.calculate_outputs(
                inputs if index == 0 else self.layers[index - 1].outputs
            )
        return self.layers[-1].outputs
