from ..types import FloatArray, LossFunction


def sum_squared_errors(
    derived_activation: float | FloatArray,
    output_value: float | FloatArray,
    target_value: float | FloatArray,
) -> float | FloatArray:
    return derived_activation * (output_value - target_value)


def categorical_cross_entropy_with_softmax(
    output_value: float | FloatArray, target_value: float | FloatArray
) -> float | FloatArray:
    return output_value - target_value


def loss(
    function: LossFunction,
    output_value: float | FloatArray,
    target_value: float | FloatArray,
    derived_activation: float | FloatArray | None = None,
) -> float | FloatArray:
    match function:
        case "CCE-SOFTMAX":
            return categorical_cross_entropy_with_softmax(output_value, target_value)
        case "SSE":
            if derived_activation is None:
                raise ValueError("derived_activation needed for SSE")
            return sum_squared_errors(derived_activation, output_value, target_value)
        case _:
            raise ValueError(f"Unsupported loss function: {function}")
