from typing import Tuple, Callable, List, Optional

from ..model import Model
from ..types import Array


# TODO: more specific types?
InT = List[Array]
OutT = List[Array]


def with_list2array(layer: Model, *, pad: int = 0) -> Model[InT, OutT]:
    return Model(
        f"with_list2array-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        attrs={"pad": pad},
    )


def forward(model: Model[InT, OutT], Xs: InT, is_train: bool) -> Tuple[OutT, Callable]:
    layer = model.layers[0]
    pad = model.get_attr("pad")
    lengths = layer.ops.asarray([len(seq) for seq in Xs])
    Xf = layer.ops.flatten(Xs, pad=pad)
    Yf, get_dXf = layer(Xf, is_train)

    def backprop(dYs: OutT) -> InT:
        dYf = layer.ops.flatten(dYs, pad=pad)
        dXf = get_dXf(dYf)
        return layer.ops.unflatten(dXf, lengths, pad=pad)

    return layer.ops.unflatten(Yf, lengths, pad=pad), backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
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
