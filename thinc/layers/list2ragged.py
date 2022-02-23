from typing import Tuple, List, Callable

from ..model import Model
from ..config import registry
from ..types import ArrayXd_Concatenable, Ragged


InT = List[ArrayXd_Concatenable]
OutT = Ragged


@registry.layers("list2ragged.v1")
def list2ragged() -> Model[InT, OutT]:
    """Transform sequences to ragged arrays if necessary and return the ragged
    array. If sequences are already ragged, do nothing. A ragged array is a
    tuple (data, lengths), where data is the concatenated data.
    """
    return Model("list2ragged", forward)


def forward(model: Model[InT, OutT], Xs: InT, is_train: bool) -> Tuple[OutT, Callable]:
    def backprop(dYr: OutT) -> InT:
        return model.ops.unflatten(dYr.data, dYr.lengths)  # type: ignore

    lengths = model.ops.asarray1i([len(x) for x in Xs])
    return Ragged(model.ops.flatten(Xs), lengths), backprop
