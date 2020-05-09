from typing import Tuple, List, Callable, cast

from ..model import Model
from ..config import registry
from ..types import Floats2d, Ragged


InT = List[Floats2d]
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
        # TODO: Unhack
        return model.ops.unflatten(cast(Floats2d, dYr.data), dYr.lengths)

    lengths = model.ops.asarray1i([len(x) for x in Xs])
    return Ragged(model.ops.flatten(Xs), lengths), backprop
