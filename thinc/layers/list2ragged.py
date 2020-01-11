from typing import Tuple, List, Callable, cast

from ..model import Model
from ..config import registry
from ..types import Array2d, Ragged


InT = List[Array2d]
OutT = Ragged


@registry.layers("list2ragged.v0")
def list2ragged() -> Model[InT, OutT]:
    """Transform sequences to ragged arrays if necessary and return the ragged
    array. If sequences are already ragged, do nothing. A ragged array is a
    tuple (data, lengths), where data is the concatenated data.
    """
    return Model("list2ragged", forward)


def forward(model: Model[InT, OutT], Xs: InT, is_train: bool) -> Tuple[OutT, Callable]:
    def backprop(dYr: OutT) -> InT:
        return cast(InT, model.ops.unflatten(dYr.data, dYr.lengths))

    lengths = model.ops.asarray([len(x) for x in Xs], dtype="i")
    return Ragged(model.ops.flatten(Xs), lengths), backprop
