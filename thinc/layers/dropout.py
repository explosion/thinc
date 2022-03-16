from typing import Tuple, Callable, List, TypeVar, cast, Union

from ..model import Model
from ..config import registry
from ..types import ArrayXd, Ragged, Padded


InT = TypeVar("InT", bound=Union[ArrayXd, List[ArrayXd], Ragged, Padded])
InT_co = TypeVar("InT_co", bound=Union[ArrayXd, List[ArrayXd], Ragged, Padded], covariant=True)


@registry.layers("Dropout.v1")
def Dropout(rate: float = 0.0) -> Model[InT_co, InT_co]:
    """Help prevent overfitting by adding a random distortion to the input data
    during training.  Specifically, cells of the input are zeroed with
    probability determined by the `rate` argument.
    """
    return Model("dropout", forward, attrs={"dropout_rate": rate, "is_enabled": True})


def forward(
    model: Model[InT_co, InT_co], X: InT, is_train: bool
) -> Tuple[InT_co, Callable]:
    rate = model.attrs["dropout_rate"]
    is_enabled = model.attrs["is_enabled"] and is_train
    if rate == 0 or not is_enabled:
        unchanged_return_value, backprop = X, lambda dY: dY
        return_value = cast(InT_co, unchanged_return_value)
    elif isinstance(X, Ragged):
        ragged_return_value, backprop = _dropout_ragged(model, X, is_train)
        return_value = cast(InT_co, ragged_return_value)
    elif isinstance(X, Padded):
        padded_return_value, backprop = _dropout_padded(model, X, is_train)
        return_value = cast(InT_co, padded_return_value)
    elif isinstance(X, List):
        list_return_value, backprop = _dropout_lists(model, X, is_train)
        return_value = cast(InT_co, list_return_value)
    else:
        array_return_value, backprop = _dropout_array(model, cast(ArrayXd, X), is_train)
        return_value = cast(InT_co, array_return_value)
    return return_value, backprop


def _dropout_array(
    model: Model[InT_co, InT_co], X: ArrayXd, is_train: bool
) -> Tuple[ArrayXd, Callable]:
    rate = model.attrs["dropout_rate"]
    mask = model.ops.get_dropout_mask(X.shape, rate)

    def backprop(dY: ArrayXd) -> ArrayXd:
        return dY * mask

    return cast(ArrayXd, X * mask), backprop


def _dropout_padded(
    model: Model[InT_co, InT_co], Xp: Padded, is_train: bool
) -> Tuple[Padded, Callable]:
    X = Xp.data
    mask = model.ops.get_dropout_mask(X.shape, model.attrs["dropout_rate"])
    Y = X * mask

    def backprop(dYp: Padded) -> Padded:
        return Padded(dYp.data * mask, dYp.size_at_t, dYp.lengths, dYp.indices)

    return Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices), backprop


def _dropout_ragged(
    model: Model[InT_co, InT_co], Xr: Ragged, is_train: bool
) -> Tuple[Ragged, Callable]:
    X = Xr.data
    lengths = Xr.lengths
    mask = model.ops.get_dropout_mask(X.shape, model.attrs["dropout_rate"])
    Y = X * mask

    def backprop(dYr: Ragged) -> Ragged:
        return Ragged(dYr.data * mask, dYr.lengths)

    return Ragged(Y, lengths), backprop


def _dropout_lists(
    model: Model[InT_co, InT_co], Xs: List[ArrayXd], is_train: bool
) -> Tuple[List[ArrayXd], Callable]:
    rate = model.attrs["dropout_rate"]
    masks = [model.ops.get_dropout_mask(X.shape, rate) for X in Xs]
    Ys = [X * mask for X, mask in zip(Xs, masks)]

    def backprop(dYs: List[ArrayXd]) -> List[ArrayXd]:
        return [dY * mask for dY, mask in zip(dYs, masks)]

    return Ys, backprop
