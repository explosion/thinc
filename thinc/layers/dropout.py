from typing import Tuple, Callable, List, Union

from ..model import Model
from ..types import Array, Ragged, Padded


InT = Union[Array, List[Array], Ragged, Padded]
OutT = Union[Array, List[Array], Ragged, Padded]


def Dropout(rate: float = 0.0) -> Model[InT, OutT]:
    """Help prevent overfitting by adding a random distortion to the input data
    during training.  Specifically, cells of the input are zeroed with
    probability determined by the `rate` argument.
    """
    return Model("dropout", forward, attrs={"rate": rate, "is_enabled": True})


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    rate = model.get_attr("rate")
    is_enabled = model.get_attr("is_enabled")
    if rate == 0 or not is_enabled:
        return X, lambda dY: dY
    elif isinstance(X, Ragged):
        return _dropout_ragged(model, X, is_train)  # type: ignore
    elif isinstance(X, Padded):
        return _dropout_padded(model, X, is_train)  # type: ignore
    elif isinstance(X, list):
        return _dropout_lists(model, X, is_train)
    else:
        return _dropout_array(model, X, is_train)


def _dropout_array(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    rate = model.get_attr("rate")
    mask = model.ops.get_dropout_mask(X.shape, rate)

    def backprop(dY: Array) -> Array:
        return dY * mask

    return X * mask, backprop


def _dropout_padded(
    model: Model[InT, OutT], Xp: Padded, is_train: bool
) -> Tuple[Padded, Callable]:
    X = Xp.data
    mask = model.ops.get_dropout_mask(X.shape, model.get_attr("rate"))
    Y = X * mask

    def backprop(dYp: Padded) -> Padded:
        return Padded(dYp.data * mask, dYp.size_at_t)

    return Padded(Y, Xp.size_at_t), backprop


def _dropout_ragged(
    model: Model, Xr: Ragged, is_train: bool
) -> Tuple[Ragged, Callable]:
    X = Xr.data
    lengths = Xr.lengths
    mask = model.ops.get_dropout_mask(X.shape, model.get_attr("rate"))
    Y = X * mask

    def backprop(dYr: Ragged) -> Ragged:
        return Ragged(dYr.data * mask, dYr.lengths)

    return Ragged(Y, lengths), backprop


def _dropout_lists(
    model: Model, Xs: List[Array], is_train: bool
) -> Tuple[List[Array], Callable]:
    rate = model.get_attr("rate")
    masks = [model.ops.get_dropout_mask(X.shape, rate) for X in Xs]
    Ys = [X * mask for X, mask in zip(Xs, masks)]

    def backprop(dYs: List[Array]) -> List[Array]:
        return [dY * mask for dY, mask in zip(dYs, masks)]

    return Ys, backprop
