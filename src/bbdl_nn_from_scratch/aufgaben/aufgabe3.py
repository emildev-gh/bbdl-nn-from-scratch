from ..neuralnetwork import NeuralNetwork


def main():
    nn = NeuralNetwork(2, [1], 2, ("relu", "softmax"))
