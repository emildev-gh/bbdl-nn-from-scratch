from .layer import Layer
from .types import ActivationConfig, LossFunction


class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_size: list[int],
        output_size: int,
        activation_config: ActivationConfig,
        loss_function: LossFunction,
    ) -> None:
        self.layers = []
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
            Layer(
                "output", activation_config[1], self.layers[-1].num_nodes, output_size
            )
        )
        self.loss_function = loss_function

    def forward(self, inputs: list[float]) -> list[float]:
        for index, layer in enumerate(self.layers):
            layer.calc_outputs(inputs if index == 0 else self.layers[index - 1].outputs)
        return self.layers[-1].outputs
