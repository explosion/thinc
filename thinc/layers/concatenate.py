from typing import Tuple, Callable, Optional, TypeVar, cast

from ..model import Model
from ..config import registry
from ..types import Array2d
from ..util import get_width
from .noop import noop
from ..types import XY_XY_OutT


InT = TypeVar("InT", bound=Array2d)
OutT = TypeVar("OutT", bound=Array2d)


@registry.layers("concatenate.v1")
def concatenate(*layers: Model) -> Model[InT, XY_XY_OutT]:
    """Compose two or more models `f`, `g`, etc, such that their outputs are
    concatenated, i.e. `concatenate(f, g)(x)` computes `hstack(f(x), g(x))`.
    Also supports chaining more than 2 layers.
    """
    if not layers:
        return cast(Model[InT, XY_XY_OutT], noop())
    elif len(layers) == 1:
        return layers[0]
    elif layers[0]._func is forward:
        layers[0].layers.extend(layers[1:])
        return layers[0]

    return Model(
        "|".join(layer.name for layer in layers),
        forward,
        init=init,
        dims={"nO": None, "nI": None},
        layers=layers,
    )


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Ys, callbacks = zip(*[layer(X, is_train=is_train) for layer in model.layers])
    if isinstance(Ys[0], list):
        return _list_forward(model, X, Ys, callbacks, is_train)
    else:
        return _array_forward(model, X, Ys, callbacks, is_train)


def _array_forward(
    model: Model[InT, OutT], X, Ys, callbacks, is_train: bool
) -> Tuple[OutT, Callable]:
    widths = [Y.shape[1] for Y in Ys]
    output = model.ops.xp.hstack(Ys)

    def backprop(d_output: OutT) -> InT:
        dY = model.ops.xp.ascontiguousarray(d_output[:, : widths[0]])
        dX = callbacks[0](dY)
        start = widths[0]
        for bwd, width in zip(callbacks[1:], widths[1:]):
            dY = model.ops.xp.ascontiguousarray(d_output[:, start : start + width])
            dX += bwd(dY)
            start += width
        return dX

    return output, backprop


def _list_forward(
    model: Model[InT, OutT], X, Ys, callbacks, is_train: bool
) -> Tuple[OutT, Callable]:
    lengths = model.ops.asarray1i([len(x) for x in X])
    Ys = [model.ops.xp.concatenate(Y, axis=0) for Y in Ys]
    widths = [Y.shape[1] for Y in Ys]
    output = model.ops.xp.hstack(Ys)
    output = model.ops.unflatten(output, lengths)

    def backprop(d_output: OutT) -> InT:
        d_output = model.ops.xp.concatenate(d_output, axis=0)
        dY = model.ops.as_contig(d_output[:, : widths[0]])
        # We want to generalize unflatten later.
        dY = model.ops.asarray(model.ops.unflatten(dY, lengths))  # type: ignore
        dX = callbacks[0](dY)
        start = widths[0]
        for bwd, width in zip(callbacks[1:], widths[1:]):
            dY = model.ops.as_contig(d_output[:, start : start + width])
            dY = model.ops.asarray(model.ops.unflatten(dY, lengths))  # type: ignore
            dX += bwd(dY)
            start += width
        return dX

    return output, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> Model[InT, OutT]:
    if X is not None:
        X_width = get_width(X)
        model.set_dim("nI", X_width)
        for layer in model.layers:
            layer.set_dim("nI", X_width)
    for layer in model.layers:
        layer.initialize(X=X, Y=Y)
    if None not in [layer.has_dim("nO") for layer in model.layers]:
        model.set_dim("nO", sum(layer.get_dim("nO") for layer in model.layers))
    return model
