from typing import Tuple, Callable, Optional, Union, List

from ..model import Model
from ..config import registry
from ..types import Array, Ragged, Padded


InT = Union[List[Array], Ragged, Padded, Array]


@registry.layers("Residual.v0")
def Residual(layer: Model[InT, InT]) -> Model[InT, InT]:
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


def forward(model: Model[InT, InT], X: InT, is_train: bool) -> Tuple[InT, Callable]:
    def backprop(d_output: InT) -> InT:
        dX = backprop_layer(d_output)
        if isinstance(d_output, list):
            return [d_output[i] + dX[i] for i in range(len(d_output))]
        elif isinstance(d_output, Ragged):
            return Ragged(d_output.data + dX.data, dX.lengths)
        elif isinstance(X, Padded):
            return Padded(d_output.data + dX.data, dX.lengths)
        else:
            return d_output + dX

    Y, backprop_layer = model.layers[0](X, is_train)
    if isinstance(X, list):
        return [X[i] + Y[i] for i in range(len(X))], backprop
    elif isinstance(X, Ragged):
        return Ragged(X.data + Y.data, X.lengths), backprop
    elif isinstance(X, Padded):
        return Padded(X.data + Y.data, X.size_at_t), backprop
    else:
        return X + Y, backprop


def init(
    model: Model[InT, InT], X: Optional[InT] = None, Y: Optional[InT] = None
) -> None:
    model.layers[0].initialize(X=X, Y=Y)
    model.set_dim("nO", model.layers[0].get_dim("nO"))
    model.set_dim("nI", model.layers[0].get_dim("nI"))
