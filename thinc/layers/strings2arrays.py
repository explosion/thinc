from typing import Callable, List, Sequence, Tuple

from murmurhash import hash_unicode

from ..config import registry
from ..model import Model
from ..types import Ints2d

InT = Sequence[Sequence[str]]
OutT = List[Ints2d]


@registry.layers("strings2arrays.v1")
def strings2arrays() -> Model[InT, OutT]:
    """Transform a sequence of string sequences to a list of arrays."""
    return Model("strings2arrays", forward)


def forward(model: Model[InT, OutT], Xs: InT, is_train: bool) -> Tuple[OutT, Callable]:
    hashes = model.ops.asarray2i(
        [[hash_unicode(word) for word in X] for X in Xs], dtype="int32"
    )
    hash_arrays = [model.ops.asarray1i(h, dtype="uint64") for h in hashes]
    arrays = [model.ops.reshape2i(array, -1, 1) for array in hash_arrays]

    def backprop(dX: OutT) -> InT:
        return []

    return arrays, backprop
