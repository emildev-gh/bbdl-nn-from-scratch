from ..neuralnetwork import NeuralNetwork
from ..types import TrainingData


def main():
    training_data: TrainingData = [
        ([0.0, 0.0], [1.0, 0.0]),
        ([0.0, 1.0], [0.0, 1.0]),
        ([1.0, 0.0], [0.0, 1.0]),
        ([1.0, 1.0], [1.0, 0.0]),
    ]  # [input][expected output (interpreted as classes: [1, 0] -> "0"; [0, 1] -> "1")]
    nn = NeuralNetwork(2, [4], 2, ("relu", "softmax"), "CCE-SOFTMAX")

    nn.train(training_data, epochs=10000)
    nn.printNetwork()
