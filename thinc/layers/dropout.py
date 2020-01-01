from typing import Tuple, Callable, TypeVar, List, Union

from ..model import Model
from ..types import Array


InputTypeArray = TypeVar("InputTypeArray", bound=Array)
InputLengths = TypeVar("InputLengths", bound=Array)
InputTypeList = List[InputTypeArray]
InputTypeRagged = Tuple[InputTypeArray, InputLengths]
InputType = Union[InputTypeArray, InputTypeList, InputTypeRagged]
OutputTypeArray = TypeVar("OutputTypeArray", bound=Array)
OutputLengths = TypeVar("OutputLengths", bound=Array)
OutputTypeList = List[OutputTypeArray]
OutputTypeRagged = Tuple[OutputTypeArray, OutputLengths]
OutputType = Union[OutputTypeArray, OutputTypeList, OutputTypeRagged]


def Dropout(rate: float = 0.0) -> Model:
    return Model("dropout", forward, attrs={"rate": rate, "is_enabled": True})


def forward(model: Model, X: InputType, is_train: bool) -> Tuple[OutputType, Callable]:
    rate = model.get_attr("rate")
    is_enabled = model.get_attr("is_enabled")
    if rate == 0 or not is_enabled:
        return X, lambda dY: dY
    elif isinstance(X, tuple) and len(X) == 2:
        return _dropout_ragged(model, X, is_train)  # type: ignore
    elif isinstance(X, list):
        return _dropout_lists(model, X, is_train)
    else:
        return _dropout_array(model, X, is_train)


def _dropout_array(
    model: Model, X: InputTypeArray, is_train: bool
) -> Tuple[OutputTypeArray, Callable]:
    rate = model.get_attr("rate")
    mask = model.ops.get_dropout_mask(X.shape, rate)

    def backprop(dY: OutputTypeArray) -> InputTypeArray:
        return dY * mask

    return X * mask, backprop


def _dropout_ragged(
    model: Model, X_lengths: InputTypeRagged, is_train: bool
) -> Tuple[OutputTypeRagged, Callable]:
    X, lengths = X_lengths
    mask = model.ops.get_dropout_mask(X.shape, model.get_attr("rate"))
    Y = X * mask

    def backprop(dY_lengths: OutputTypeRagged) -> InputTypeRagged:
        dY, lengths = dY_lengths
        return (dY * mask), lengths

    return (Y, lengths), backprop


def _dropout_lists(
    model: Model, Xs: InputTypeList, is_train: bool
) -> Tuple[OutputTypeList, Callable]:
    rate = model.get_attr("rate")
    masks = [model.ops.get_dropout_mask(X.shape, rate) for X in Xs]
    Ys = [X * mask for X, mask in zip(Xs, masks)]

    def backprop(dYs: OutputTypeList) -> InputTypeList:
        return [dY * mask for dY, mask in zip(dYs, masks)]

    return Ys, backprop
