from typing import Tuple, Callable, Optional, Union, List, TypeVar

from ..model import Model
from ..types import Array


# TODO: fix
InputValue = TypeVar("InputValue", bound=Array)
InputLengths = TypeVar("InputLengths", bound=Array)
InputType = Union[Tuple[InputValue, InputLengths], List[InputValue], InputValue]
OutputValue = TypeVar("OutputValue", bound=Array)
OutputLengths = TypeVar("OutputLengths", bound=Array)
OutputType = Union[Tuple[OutputValue, OutputLengths], List[OutputValue], OutputValue]


def Residual(layer: Model) -> Model:
    return Model(
        "residual",
        forward,
        init=init,
        layers=[layer],
        dims={
            "nO": layer.get_dim("nO") if layer.has_dim("nO") else None,
            "nI": layer.get_dim("nI") if layer.has_dim("nI") else None
        },
    )


def forward(model: Model, X: InputType, is_train: bool) -> Tuple[OutputType, Callable]:
    y, bp_y = model.layers[0].begin_update(X)
    if isinstance(X, list):
        output = [X[i] + y[i] for i in range(len(X))]
    elif isinstance(X, tuple) and isinstance(y, tuple) and len(X) == 2:
        # Handle case where we have (data, lengths) tuple
        output = (X[0] + y[0], y[1])  # type: ignore
    else:
        output = X + y

    def backprop(d_output: OutputType) -> InputType:
        dX = bp_y(d_output)
        if isinstance(d_output, list) or isinstance(d_output, tuple):
            return [d_output[i] + dX[i] for i in range(len(d_output))]
        else:
            return d_output + dX

    return output, backprop


def init(
    model: Model, X: Optional[InputType] = None, Y: Optional[OutputType] = None
) -> None:
    model.layers[0].initialize(X=X, Y=Y)
    model.set_dim("nO", model.layers[0].get_dim("nO"))
    model.set_dim("nI", model.layers[0].get_dim("nI"))
