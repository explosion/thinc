from typing import Tuple, Callable, List, Optional, TypeVar, Union, cast

from ..model import Model
from ..config import registry
from ..types import Array2d, Padded, Ragged


ValT = TypeVar("ValT", bound=Array2d)
SeqT = TypeVar("SeqT", bound=Union[Padded, Ragged, List[Array2d], Array2d])


@registry.layers("with_array.v0")
def with_array(layer: Model[ValT, ValT], pad: int = 0) -> Model[SeqT, SeqT]:
    """Transform sequence data into a contiguous 2d array on the way into and
    out of a model. Handles a variety of sequence types: lists, padded and ragged.
    If the input is a 2d array, it is passed through unchanged.
    """
    return Model(
        f"with_array-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        attrs={"pad": pad},
    )


def forward(model: Model[SeqT, SeqT], Xseq: SeqT, is_train: bool):
    if isinstance(Xseq, Ragged):
        return _ragged_forward(
            cast(Model[Ragged, Ragged], model), cast(Ragged, Xseq), is_train
        )
    elif isinstance(Xseq, Padded):
        return _padded_forward(
            cast(Model[Padded, Padded], model), cast(Padded, Xseq), is_train
        )
    elif not isinstance(Xseq, (list, tuple)):
        return model.layers[0](Xseq, is_train)
    else:
        return _list_forward(
            cast(Model[List[Array2d], List[Array2d]], model), Xseq, is_train
        )


def init(
    model: Model[SeqT, SeqT], X: Optional[SeqT] = None, Y: Optional[SeqT] = None
) -> Model[SeqT, SeqT]:
    layer: Model[Array2d, Array2d] = model.layers[0]
    layer.initialize(
        X=_get_array(model, X) if X is not None else X,
        Y=_get_array(model, Y) if Y is not None else Y,
    )
    return model


def _get_array(model, X: SeqT) -> Array2d:
    if isinstance(X, Ragged):
        return X.data
    elif isinstance(X, Padded):
        return X.data.reshape((X.data.shape[0] * X.data.shape[1], X.data.shape[2]))
    elif not isinstance(X, (list, tuple)):
        return cast(Array2d, X)
    else:
        return model.ops.flatten(X)


def _list_forward(
    model: Model[List[Array2d], List[Array2d]], Xs: List[Array2d], is_train: bool
) -> Tuple[List[Array2d], Callable]:
    layer = model.layers[0]
    pad = model.get_attr("pad")
    lengths = layer.ops.asarray([len(seq) for seq in Xs])
    Xf = layer.ops.flatten(Xs, pad=pad)
    Yf, get_dXf = layer(Xf, is_train)

    def backprop(dYs: List[Array2d]) -> List[Array2d]:
        dYf = layer.ops.flatten(dYs, pad=pad)
        dXf = get_dXf(dYf)
        return layer.ops.unflatten(dXf, lengths, pad=pad)

    return layer.ops.unflatten(Yf, lengths, pad=pad), backprop


def _ragged_forward(
    model: Model[Ragged, Ragged], Xr: Ragged, is_train: bool
) -> Tuple[Ragged, Callable]:
    layer: Model[Array2d, Array2d] = model.layers[0]
    Y, get_dX = layer(Xr.data, is_train)

    def backprop(dYr: Ragged) -> Ragged:
        return Ragged(get_dX(dYr.data), dYr.lengths)

    return Ragged(Y, Xr.lengths), backprop


def _padded_forward(
    model: Model[Padded, Padded], Xp: Padded, is_train: bool
) -> Tuple[Padded, Callable]:
    layer: Model[Array2d, Array2d] = model.layers[0]
    X = Xp.data.reshape((Xp.data.shape[0] * Xp.data.shape[1], Xp.data.shape[2]))
    Y2d, get_dX = layer(X, is_train)
    Y = Y2d.reshape((Xp.data.shape[0], Xp.data.shape[1], Y2d.shape[1]))

    def backprop(dYp: Padded) -> Padded:
        dY = dYp.data.reshape(
            (dYp.data.shape[0] * dYp.data.shape[1], dYp.data.shape[2])
        )
        dX2d = get_dX(dY)
        dX = dX2d.reshape((dYp.data.shape[0], dYp.data.shape[1], dX2d.data.shape[1]))
        return Padded(dX, dYp.size_at_t, dYp.lengths, dYp.indices)

    return Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices), backprop
