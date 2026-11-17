from ..types import LossFunction


def sum_squared_errors(
    derived_activation: float, output_value: float, target_value: float
) -> float:
    return derived_activation * (output_value - target_value)


def categorical_cross_entropy_with_softmax(output_value: float, target_value: float):
    return output_value - target_value


def loss(
    function: LossFunction,
    output_value: float,
    target_value: float,
    derived_activation: float | None = None,
) -> float:
    match function:
        case "CCE-SOFTMAX":
            return categorical_cross_entropy_with_softmax(output_value, target_value)
        case "SSE":
            if derived_activation is None:
                raise ValueError("derived_activation needed for SSE")
            return sum_squared_errors(derived_activation, output_value, target_value)
        case _:
            raise ValueError(f"Unsupported loss function: {function}")
