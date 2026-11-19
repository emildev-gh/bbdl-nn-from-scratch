from ..neuralnetwork import NeuralNetwork


def main():
    training_data = [
        ([0.0, 0.0], [1.0, 0.0]),
        ([0.0, 1.0], [0.0, 1.0]),
        ([1.0, 0.0], [0.0, 1.0]),
        ([1.0, 1.0], [1.0, 0.0]),
    ]  # [input][expected output (interpreted as classes: [1, 0] -> "0"; [0, 1] -> "1")]
    nn = NeuralNetwork(2, [4], 2, ("relu", "softmax"), "CCE-SOFTMAX")
