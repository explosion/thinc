from typing import Tuple, Callable, List, TypeVar, Any

from ..model import Model
from ..config import registry
from ..types import ArrayXd, Ragged, Padded


InT = TypeVar("InT")
ArrayT = TypeVar("ArrayT", bound=ArrayXd)


@registry.layers("Dropout.v1")
def Dropout(rate: float = 0.0) -> Model[InT, InT]:
    """Help prevent overfitting by adding a random distortion to the input data
    during training.  Specifically, cells of the input are zeroed with
    probability determined by the `rate` argument.
    """
    return Model("dropout", forward, attrs={"dropout_rate": rate, "is_enabled": True})


# We're getting type hell here, I think because of the instance checks?
# It's sort of painful, because I think this confused the types of other
# layers that are trying to use dropout.
# I've relaxed the types for now, but it'd be good to understand what's wrong
# here.
def forward(model: Model, X, is_train: bool) -> Tuple[Any, Callable]:
    rate = model.attrs["dropout_rate"]
    is_enabled = model.attrs["is_enabled"] and is_train
    if rate == 0 or not is_enabled:
        return X, lambda dY: dY
    elif isinstance(X, Ragged):
        return _dropout_ragged(model, X, is_train)
    elif isinstance(X, Padded):
        return _dropout_padded(model, X, is_train)
    elif isinstance(X, list):
        return _dropout_lists(model, X, is_train)
    else:
        return _dropout_array(model, X, is_train)


def _dropout_array(
    model: Model[ArrayT, ArrayT], X: ArrayT, is_train: bool
) -> Tuple[ArrayT, Callable]:
    rate = model.attrs["dropout_rate"]
    mask = model.ops.get_dropout_mask(X.shape, rate)

    def backprop(dY: ArrayT) -> ArrayT:
        return dY * mask

    return X * mask, backprop


def _dropout_padded(
    model: Model, Xp: Padded, is_train: bool
) -> Tuple[Padded, Callable]:
    X = Xp.data
    mask = model.ops.get_dropout_mask(X.shape, model.attrs["dropout_rate"])
    Y = X * mask

    def backprop(dYp: Padded) -> Padded:
        return Padded(dYp.data * mask, dYp.size_at_t, dYp.lengths, dYp.indices)

    return Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices), backprop


def _dropout_ragged(
    model: Model, Xr: Ragged, is_train: bool
) -> Tuple[Ragged, Callable]:
    X = Xr.data
    lengths = Xr.lengths
    mask = model.ops.get_dropout_mask(X.shape, model.attrs["dropout_rate"])
    Y = X * mask

    def backprop(dYr: Ragged) -> Ragged:
        return Ragged(dYr.data * mask, dYr.lengths)

    return Ragged(Y, lengths), backprop


def _dropout_lists(
    model: Model[ArrayT, ArrayT], Xs: List[ArrayT], is_train: bool
) -> Tuple[List[ArrayT], Callable]:
    rate = model.attrs["dropout_rate"]
    masks = [model.ops.get_dropout_mask(X.shape, rate) for X in Xs]
    Ys = [X * mask for X, mask in zip(Xs, masks)]

    def backprop(dYs: List[ArrayT]) -> List[ArrayT]:
        return [dY * mask for dY, mask in zip(dYs, masks)]

    return Ys, backprop
