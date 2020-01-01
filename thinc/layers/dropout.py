from typing import Tuple, Callable, TypeVar, List

from ..model import Model
from ..types import Array


# TODO: fix
InputType = TypeVar("InputType", bound=Array)
OutputType = TypeVar("OutputType", bound=Array)


def Dropout(rate: float = 0.0) -> Model:
    return Model("dropout", forward, attrs={"rate": rate, "is_enabled": True})


def forward(model: Model, X: InputType, is_train: bool) -> Tuple[OutputType, Callable]:
    rate = model.get_attr("rate")
    is_enabled = model.get_attr("is_enabled")
    if rate == 0 or not is_enabled:
        return X, lambda dY: dY
    elif isinstance(X, tuple) and len(X) == 2:
        return _dropout_ragged(model, X, is_train)
    elif isinstance(X, list):
        return _dropout_lists(model, X, is_train)
    else:
        return _dropout_array(model, X, is_train)


def _dropout_array(model: Model, X: Array, is_train: bool):
    rate = model.get_attr("rate")
    mask = model.ops.get_dropout_mask(X.shape, rate)

    def backprop(dY: Array):
        return dY * mask
    
    return X * mask, backprop


def _dropout_ragged(model: Model, X_lengths: Tuple[Array, Array], is_train: bool):
    rate = model.get_attr("rate")
    X, lengths = X_lengths
    mask = model.ops.get_dropout_mask(X.shape, model.get_attr("rate"))
    Y = X * mask

    def backprop(dY_lengths: Tuple[Array, Array]):
        dY, lengths = dY_lengths
        return (dY * mask), lengths
    
    return (Y, lengths), backprop


def _dropout_lists(model: Model, Xs: List[Array], is_train: bool):
    rate = model.get_attr("rate")
    masks = [model.ops.get_dropout_mask(X.shape, rate) for X in Xs]
    Ys = [X * mask for X, mask in zip(Xs, masks)]

    def backprop(dYs: List[Array]):
        return [dY * mask for dY, mask in zip(dYs, masks)]
    
    return Ys, backprop
