from typing import Tuple, Callable, Optional, Union, List, TypeVar

from ..model import Model
from ..types import Array


# TODO: fix
InputValue = TypeVar("InputValue", bound=Array)
InputLengths = TypeVar("InputLengths", bound=Array)
InT = Union[Tuple[InputValue, InputLengths], List[InputValue], InputValue]
OutputValue = TypeVar("OutputValue", bound=Array)
OutputLengths = TypeVar("OutputLengths", bound=Array)
OutT = Union[Tuple[OutputValue, OutputLengths], List[OutputValue], OutputValue]


def Residual(layer: Model) -> Model:
    return Model(
        "residual",
        forward,
        init=init,
        layers=[layer],
        dims={
            "nO": layer.get_dim("nO") if layer.has_dim("nO") else None,
            "nI": layer.get_dim("nI") if layer.has_dim("nI") else None,
        },
    )


def forward(model: Model, X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y: OutT
    Y, backprop_layer = model.layers[0](X, is_train)
    if isinstance(X, list):
        output = [X[i] + Y[i] for i in range(len(X))]
    elif isinstance(X, tuple) and isinstance(Y, tuple) and len(X) == 2:
        # Handle case where we have (data, lengths) tuple
        output = (X[0] + Y[0], Y[1])  # type: ignore
    else:
        output = X + Y

    def backprop(d_output: OutT) -> InT:
        dX = backprop_layer(d_output)
        if isinstance(d_output, list) or isinstance(d_output, tuple):
            return [d_output[i] + dX[i] for i in range(len(d_output))]
        else:
            return d_output + dX

    return output, backprop


def init(
    model: Model, X: Optional[InT] = None, Y: Optional[OutT] = None
) -> None:
    model.layers[0].initialize(X=X, Y=Y)
    model.set_dim("nO", model.layers[0].get_dim("nO"))
    model.set_dim("nI", model.layers[0].get_dim("nI"))
