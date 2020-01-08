from typing import Tuple, Callable, List

from ..model import Model
from ..config import registry
from ..types import Array, Ragged, FloatsNd


InT = Ragged
OutT = List[FloatsNd]


@registry.layers("ragged2list.v0")
def ragged2list() -> Model[InT, OutT]:
    """Transform sequences from a ragged format into lists."""
    return Model("ragged2list", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    lengths = Xr.lengths

    def backprop(dXs: OutT) -> InT:
        return Ragged(model.ops.flatten(dXs, pad=0), lengths)

    return model.ops.unflatten(Xr.data, Xr.lengths), backprop
