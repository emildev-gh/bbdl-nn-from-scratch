from time import perf_counter

from ..data import load_data
from ..neuralnetwork import NeuralNetwork


def main():
    training_data = load_data("train-00000-of-00001.parquet")
    testing_data = load_data("test-00000-of-00001.parquet")

    # Training
    nn = NeuralNetwork(784, [30], 10, ("RELU", "SOFTMAX"), "CCE-SOFTMAX")
    training_start = perf_counter()
    nn.train(training_data, epochs=1)
    training_time = perf_counter() - training_start
    print(f"Training time: {training_time:.2f} seconds")

    # Inference
    inference_start = perf_counter()
    prediction = nn.run(testing_data[0][0])
    inference_time = perf_counter() - inference_start
    print(f"Inference time (one image): {inference_time:.6f} seconds")
    print(prediction)
