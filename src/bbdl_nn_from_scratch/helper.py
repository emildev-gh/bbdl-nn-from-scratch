from .types import InferenceData, TrainingData


def extract_input_data(training_data: TrainingData) -> InferenceData:
    return [inputs for inputs, _ in training_data]


def extract_target_data(training_data: TrainingData) -> InferenceData:
    return [targets for _, targets in training_data]
