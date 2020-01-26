from typing import Tuple, List, Callable, Sequence
from murmurhash import hash_unicode

from ..model import Model
from ..config import registry
from ..types import Array2d, Array


InT = Sequence[str]
OutT = List[Array2d]


@registry.layers("strings2arrays.v0")
def strings2arrays() -> Model[InT, OutT]:
    """Transform a sequence of strings to a list of arrays."""
    return Model("strings2arrays", forward)


def forward(model: Model[InT, OutT], Xs: InT, is_train: bool) -> Tuple[OutT, Callable]:
    hashes = [[hash_unicode(word) for word in X] for X in Xs]
    hash_arrays: List[Array] = [model.ops.asarray(h, dtype="uint64") for h in hashes]
    arrays = [model.ops.reshape2f(array, -1, 1) for array in hash_arrays]

    def backprop(dX: OutT) -> InT:
        return []

    return arrays, backprop
