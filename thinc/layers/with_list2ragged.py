from typing import Tuple, Callable, List, Optional, TypeVar

from ..model import Model
from ..types import Array

InputType = TypeVar("InputType", bound=List[Array])
OutputType = TypeVar("OutputType", bound=List[Array])


def with_list2array(layer: Model, *, pad: int = 0) -> Model:
    return Model(
        f"with_list2array-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        attrs={"pad": pad},
    )


def forward(
    model: Model, Xs: InputType, is_train: bool
) -> Tuple[OutputType, Callable]:
    layer = model.layers[0]
    pad = model.get_attr("pad")
    lengths = layer.ops.asarray([len(seq) for seq in Xs])
    Xf = layer.ops.flatten(Xs, pad=pad)
    Yf, get_dXf = layer(Xf, is_train)

    def backprop(dYs: OutputType) -> InputType:
        dYf = layer.ops.flatten(dXs, pad=pad)
        dXf = get_dXf(dYf)
        return layer.ops.unflatten(dXf, lengths, pad=pad)

    return layer.ops.unflatten(Yf, lengths, pad=pad), backprop


def init(
    model: Model, X: Optional[InputType] = None, Y: Optional[OutputType] = None
) -> None:
    layer = model.layers[0]
    pad = model.get_attr("pad")
    if X is not None:
        Xflat = layer.ops.flatten(X, pad=pad)
    else:
        Xflat = None
    if Y is not None:
        Yflat = layer.ops.flatten(Y, pad=pad)
    else:
        Yflat = None
    layer.initialize(X=Xflat, Y=Yflat)
