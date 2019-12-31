from typing import Tuple, Callable, List, Optional, TypeVar

from ..model import Model
from ..types import Array

InputType = TypeVar("InputType", bound=List[Array])
OutputType = TypeVar("OutputType", bound=List[Array])


def with_flatten(layer: Model, *, pad: int = 0) -> Model:
    return Model(
        f"with_flatten-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        attrs={"pad": pad},
        dims={"nO": layer.get_dim("nO"), "nI": layer.get_dim("nI")},
    )


def forward(
    model: Model, seqs_in: InputType, is_train: bool
) -> Tuple[OutputType, Callable]:
    layer = model.layers[0]
    pad = model.get_attr("pad")
    lengths = layer.ops.asarray([len(seq) for seq in seqs_in])
    X, bp_layer = layer.begin_update(layer.ops.flatten(seqs_in, pad=pad))

    def backprop(d_seqs_out: OutputType) -> InputType:
        d_X = bp_layer(layer.ops.flatten(d_seqs_out, pad=pad))
        return layer.ops.unflatten(d_X, lengths, pad=pad)

    return layer.ops.unflatten(X, lengths, pad=pad), backprop


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
    if layer.has_dim("nI"):
        model.set_dim("nI", layer.get_dim("nI"))
    if layer.has_dim("nO"):
        model.set_dim("nO", layer.get_dim("nO"))
