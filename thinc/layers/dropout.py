from typing import Tuple, Callable, List, TypeVar, cast, Union, Sequence

from ..model import Model
from ..config import registry
from ..types import ArrayXd, Ragged, Padded


InT = TypeVar("InT", bound=Union[ArrayXd, Sequence[ArrayXd], Ragged, Padded])


@registry.layers("Dropout.v1")
def Dropout(rate: float = 0.0) -> Model[InT, InT]:
    """Help prevent overfitting by adding a random distortion to the input data
    during training.  Specifically, cells of the input are zeroed with
    probability determined by the `rate` argument.
    """
    return Model(
        "dropout",
        forward,
        attrs={
            "dropout_attr": "dropout_rate",
            "dropout_rate": rate,
            "is_enabled": True,
        },
    )


@registry.layers("Dropout.v2")
def Dropout_v2(
    rate: float = 0.0, *, dropout_attr: str = "dropout_rate"
) -> Model[InT, InT]:
    """Help prevent overfitting by adding a random distortion to the input data
    during training.  Specifically, cells of the input are zeroed with
    probability determined by the `rate` argument.

    rate: the probability of the dropout mask.
    dropout_attr: the name to use for the model attribute used to store *rate*. Different names
        can be used to enable dropout layers to be sensitive to different calls to
        *Model.set_dropout_rate()*. The default both here and in *Model.set_dropout_rate()* is
        `dropout_rate`.
    """
    return Model(
        "dropout",
        forward,
        attrs={
            "dropout_attr": dropout_attr,
            dropout_attr: rate,
            "is_enabled": True,
        },
    )


def forward(model: Model[InT, InT], X: InT, is_train: bool) -> Tuple[InT, Callable]:
    dropout_attr = model.attrs["dropout_attr"]
    rate = model.attrs[dropout_attr]
    is_enabled = model.attrs["is_enabled"] and is_train
    if rate == 0 or not is_enabled:
        return X, lambda dY: dY
    elif isinstance(X, Ragged):
        data_r, backprop = _dropout_ragged(model, X, is_train)
        return cast(InT, data_r), backprop
    elif isinstance(X, Padded):
        data_p, backprop = _dropout_padded(model, X, is_train)
        return cast(InT, data_p), backprop
    elif isinstance(X, Sequence):
        data_l, backprop = _dropout_lists(model, X, is_train)
        return cast(InT, data_l), backprop
    else:
        data_a, backprop = _dropout_array(model, cast(ArrayXd, X), is_train)
        return cast(InT, data_a), backprop


def _dropout_array(
    model: Model[InT, InT], X: ArrayXd, is_train: bool
) -> Tuple[ArrayXd, Callable]:
    dropout_attr = model.attrs["dropout_attr"]
    rate = model.attrs[dropout_attr]
    mask = model.ops.get_dropout_mask(X.shape, rate)

    def backprop(dY: ArrayXd) -> ArrayXd:
        return dY * mask

    return cast(ArrayXd, X * mask), backprop


def _dropout_padded(
    model: Model[InT, InT], Xp: Padded, is_train: bool
) -> Tuple[Padded, Callable]:
    X = Xp.data
    dropout_attr = model.attrs["dropout_attr"]
    rate = model.attrs[dropout_attr]
    mask = model.ops.get_dropout_mask(X.shape, rate)
    Y = X * mask

    def backprop(dYp: Padded) -> Padded:
        return Padded(dYp.data * mask, dYp.size_at_t, dYp.lengths, dYp.indices)

    return Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices), backprop


def _dropout_ragged(
    model: Model[InT, InT], Xr: Ragged, is_train: bool
) -> Tuple[Ragged, Callable]:
    X = Xr.data
    lengths = Xr.lengths
    dropout_attr = model.attrs["dropout_attr"]
    rate = model.attrs[dropout_attr]
    mask = model.ops.get_dropout_mask(X.shape, rate)
    Y = X * mask

    def backprop(dYr: Ragged) -> Ragged:
        return Ragged(dYr.data * mask, dYr.lengths)

    return Ragged(Y, lengths), backprop


def _dropout_lists(
    model: Model[InT, InT], Xs: Sequence[ArrayXd], is_train: bool
) -> Tuple[Sequence[ArrayXd], Callable]:
    dropout_attr = model.attrs["dropout_attr"]
    rate = model.attrs[dropout_attr]
    masks = [model.ops.get_dropout_mask(X.shape, rate) for X in Xs]
    Ys = [X * mask for X, mask in zip(Xs, masks)]

    def backprop(dYs: List[ArrayXd]) -> List[ArrayXd]:
        return [dY * mask for dY, mask in zip(dYs, masks)]

    return Ys, backprop
