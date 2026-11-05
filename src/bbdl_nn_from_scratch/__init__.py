from .neuralnetwork import NeuralNetwork


def main() -> None:
    nn = NeuralNetwork(2, [3], 2)
    print(nn.forward([2, 3]))
