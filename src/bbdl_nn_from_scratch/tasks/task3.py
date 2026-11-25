# from ..helper import build_batches
import numpy as np

from ..neuralnetwork import NeuralNetwork
from ..types import TrainingData


def main():
    training_data: TrainingData = (
        np.asarray(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=float,
        ),
        np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            dtype=float,
        ),
    )  # [input][expected output (interpreted as classes: [1, 0] -> "0"; [0, 1] -> "1")]

    # testing data batching
    # training_data_batched = build_batches(training_data)
    # print(training_data)
    # print("-------------")
    # print(training_data_batched)

    # Training
    nn = NeuralNetwork(2, [4], 2, ("RELU", "SOFTMAX"), "CCE-SOFTMAX")
    nn.train(training_data, epochs=10000)
    nn.printNetwork()

    # Inference
    # nn = NeuralNetwork.load("./models/xor.json")
    print(nn.run([0.0, 1.0]))
