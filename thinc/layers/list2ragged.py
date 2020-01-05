from typing import Tuple, List, Callable, TypeVar

from ..model import Model
from ..types import Array, Ragged


InputValue = TypeVar("InputValue", bound=Array)
InT = List[InputValue]
OutT = Ragged


def list2ragged() -> Model:
    """Transform sequences to ragged arrays if necessary. If sequences are
    already ragged, do nothing. A ragged array is a tuple (data, lengths),
    where data is the concatenated data.
    """
    return Model("list2ragged", forward)


def forward(model: Model, Xs: InT, is_train: bool) -> Tuple[OutT, Callable]:
    def backprop(dYr: OutT) -> InT:
        return model.ops.unflatten(dYr.data, dYr.lengths)

    lengths = model.ops.asarray([len(x) for x in Xs], dtype="i")
    return Ragged(model.ops.flatten(Xs), lengths), backprop
