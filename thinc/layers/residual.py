from typing import Tuple, Callable, Optional, List, TypeVar, cast, Union

from ..model import Model
from ..config import registry
from ..types import Floats1d, Floats2d, Floats3d, Floats4d, FloatsXd, Ragged, Padded


InT = TypeVar(
    "InT",
    bound=Union[
        List[Floats1d],
        List[Floats2d],
        List[Floats3d],
        List[Floats4d],
        Ragged,
        Padded,
        FloatsXd,
    ],
)
InT_co = TypeVar(
    "InT_co",
    bound=Union[
        List[Floats1d],
        List[Floats2d],
        List[Floats3d],
        List[Floats4d],
        Ragged,
        Padded,
        FloatsXd,
    ],
    covariant=True,
)


@registry.layers("residual.v1")
def residual(layer: Model[InT_co, InT_co]) -> Model[InT_co, InT_co]:
    return Model(
        f"residual({layer.name})",
        forward,
        init=init,
        layers=[layer],
        dims={
            "nO": layer.get_dim("nO") if layer.has_dim("nO") else None,
            "nI": layer.get_dim("nI") if layer.has_dim("nI") else None,
        },
    )


def forward(
    model: Model[InT_co, InT_co], X: InT, is_train: bool
) -> Tuple[InT_co, Callable]:
    def backprop(d_output: InT) -> InT:
        dX = backprop_layer(d_output)
        if isinstance(d_output, list):
            return cast(InT, [d_output[i] + dX[i] for i in range(len(d_output))])
        elif isinstance(d_output, Ragged):
            ragged_d_output = cast(Ragged, d_output)
            return cast(InT, Ragged(ragged_d_output.data + dX.data, dX.lengths))
        elif isinstance(X, Padded):
            dX.data += d_output.data  # type: ignore[union-attr]
            return dX
        else:
            return d_output + dX

    Y, backprop_layer = model.layers[0](X, is_train)
    if isinstance(X, list):
        return cast(InT_co, [X[i] + Y[i] for i in range(len(X))]), backprop
    elif isinstance(X, Ragged):
        ragged_X = cast(Ragged, X)
        return cast(InT_co, Ragged(ragged_X.data + Y.data, ragged_X.lengths)), backprop
    elif isinstance(X, Padded):
        Y.data += X.data
        return Y, backprop
    else:
        return X + Y, backprop


def init(
    model: Model[InT_co, InT_co], X: Optional[InT] = None, Y: Optional[InT] = None
) -> Model[InT_co, InT_co]:
    first_layer = model.layers[0]
    if first_layer.has_dim("nO") is None:
        first_layer.initialize(X=X, Y=Y)
    else:
        first_layer.initialize(X=X)
    if first_layer.has_dim("nO"):
        model.set_dim("nO", first_layer.get_dim("nO"))
    if first_layer.has_dim("nI"):
        model.set_dim("nI", first_layer.get_dim("nI"))
    return model
