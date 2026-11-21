from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .types import ActivationConfig, LossFunction

if TYPE_CHECKING:
    from .neuralnetwork import NeuralNetwork

# AI generated


def save_network(
    network: NeuralNetwork, path: str | Path, epochs_trained_on: int | None = None
) -> None:
    state = {
        "input_size": network.layers[0].num_inputs,
        "hidden_size": [layer.num_nodes for layer in network.layers[:-1]],
        "output_size": network.layers[-1].num_nodes,
        "activation_config": [
            network.layers[0].activation_function,
            network.layers[-1].activation_function,
        ],
        "loss_function": network.loss_function,
        "epochs_trained": epochs_trained_on,
        "layers": [
            {
                "weights": layer.weights,
                "biases": layer.biases,
            }
            for layer in network.layers
        ],
    }

    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_network(path: str | Path) -> NeuralNetwork:
    from .neuralnetwork import NeuralNetwork

    model_path = Path(path)
    state = json.loads(model_path.read_text(encoding="utf-8"))

    activation_config = cast(
        ActivationConfig,
        (
            state["activation_config"][0],
            state["activation_config"][1],
        ),
    )
    loss_function = cast(LossFunction, state["loss_function"])

    network = NeuralNetwork(
        state["input_size"],
        state["hidden_size"],
        state["output_size"],
        activation_config,
        loss_function,
    )

    saved_layers = state["layers"]
    if len(saved_layers) != len(network.layers):
        raise ValueError("model layer count does not match the network configuration")

    for layer, saved_layer in zip(network.layers, saved_layers, strict=True):
        weights = saved_layer["weights"]
        biases = saved_layer["biases"]

        if len(weights) != layer.num_nodes or any(
            len(node_weights) != layer.num_inputs for node_weights in weights
        ):
            raise ValueError("model weight shape does not match the network configuration")
        if len(biases) != layer.num_nodes:
            raise ValueError("model bias shape does not match the network configuration")

        layer.weights = weights
        layer.biases = biases

    return network
