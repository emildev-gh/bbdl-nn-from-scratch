from ..data import load_data, load_data_split
from ..helper import calc_accuracy
from ..neuralnetwork import NeuralNetwork
from ..types import TrainingData
from ..visualization import display_image


def train() -> NeuralNetwork:
    training_data, validation_data = load_data_split("train-00000-of-00001.parquet")
    print(f"Training samples: {len(training_data[0])}")
    print(f"Validation samples: {len(validation_data[0])}")

    print("starting training...")
    nn = NeuralNetwork(784, [30], 10, ("RELU", "SOFTMAX"), "CCE-SOFTMAX")
    nn.train(
        training_data=training_data,
        validation_data=validation_data,
        desired_accuracy=99,
        epochs=500,
        model_name="99desiredAccuracy500maxEpochs2.json",
    )
    return nn


def load_pretrained(model_name: str) -> NeuralNetwork:
    return NeuralNetwork.load(f"models/{model_name}")


def infere(nn: NeuralNetwork, data: TrainingData, instance: int):
    print("starting inference...")
    prediction = nn.run(data[0][instance])
    display_image((data[0][instance], data[1][instance]))
    for index, probability in enumerate(prediction):
        print(f"Class {index}: {probability:.6f}")


def test_accuracy(nn: NeuralNetwork, data: TrainingData):
    print("testing accuracy...")
    accuracy = calc_accuracy(nn, data)
    print(f"Test accuracy: {accuracy:.2f}%")


def main():
    nn = load_pretrained("99desiredAccuracy500maxEpochs.json")
    testing_data = load_data("test-00000-of-00001.parquet")
    print("---------------------------------------")
    for i in range(10):
        infere(nn, testing_data, i)
    print("---------------------------------------")
    test_accuracy(nn, testing_data)
