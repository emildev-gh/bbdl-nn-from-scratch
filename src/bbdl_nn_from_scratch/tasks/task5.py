from ..data import load_data, load_data_split
from ..neuralnetwork import NeuralNetwork
from ..visualization import display_image


def main():
    training_data, validation_data = load_data_split("train-00000-of-00001.parquet")
    print(f"Training samples: {len(training_data[0])}")
    print(f"Validation samples: {len(validation_data[0])}")

    # Training
    nn = NeuralNetwork(784, [30], 10, ("RELU", "SOFTMAX"), "CCE-SOFTMAX")
    nn.train(training_data, epochs=1)

    # Inference
    testing_data = load_data("test-00000-of-00001.parquet")
    display_image((testing_data[0][0], testing_data[1][0]))
    prediction = nn.run(testing_data[0][0])
    print(prediction)
