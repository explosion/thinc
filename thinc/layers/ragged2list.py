from typing import Tuple, Callable, TypeVar, List

from ..model import Model
from ..types import Array, Ragged


InputValue = TypeVar("InputValue", bound=Array)
InT = Ragged
OutT = List[InputValue]


def ragged2list() -> Model:
    """Transform sequences from a ragged format into lists."""
    return Model("ragged2list", forward)


def forward(model: Model, Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    lengths = Xr.lengths

    def backprop(dXs: OutT) -> InT:
        return Ragged(model.ops.flatten(dXs, pad=0), lengths)

    return model.ops.unflatten(Xr.data, Xr.lengths), backprop
