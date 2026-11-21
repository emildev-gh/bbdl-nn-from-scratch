from .types import ActivationConfig, InferenceData, LossFunction, TrainingData


def extract_input_data(training_data: TrainingData) -> InferenceData:
    return [inputs for inputs, _ in training_data]


def extract_target_data(training_data: TrainingData) -> InferenceData:
    return [targets for _, targets in training_data]


def validate_network_configuration(
    input_size: int,
    hidden_size: list[int],
    output_size: int,
    activation_config: ActivationConfig,
    loss_function: LossFunction,
) -> None:
    if input_size < 1:
        raise ValueError("input_size must be greater than 0")
    if output_size < 1:
        raise ValueError("output_size must be greater than 0")
    if any(num_nodes < 1 for num_nodes in hidden_size):
        raise ValueError("hidden layer sizes must be greater than 0")
    if hidden_size and activation_config[0] not in ("relu", "sigmoid"):
        raise ValueError("hidden activation must be 'relu' or 'sigmoid'")
    if loss_function == "SSE":
        if activation_config[1] not in ("relu", "sigmoid"):
            raise ValueError("SSE requires a 'relu' or 'sigmoid' output activation")
    elif loss_function == "CCE-SOFTMAX":
        if activation_config[1] != "softmax":
            raise ValueError("CCE-SOFTMAX requires a 'softmax' output activation")
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")
