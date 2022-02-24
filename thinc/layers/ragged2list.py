from typing import Tuple, Callable, List, TypeVar

from ..model import Model
from ..config import registry
from ..types import Ragged, ArrayXd


InT = Ragged
OutT_member_co = TypeVar("OutT_member_co", bound=ArrayXd, covariant=True)
OutT = List[OutT_member_co]


@registry.layers("ragged2list.v1")
def ragged2list() -> Model[InT, OutT]:
    """Transform sequences from a ragged format into lists."""
    return Model("ragged2list", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    lengths = Xr.lengths

    def backprop(dXs: OutT) -> InT:
        return Ragged(model.ops.flatten(dXs, pad=0), lengths)

    data: OutT = model.ops.unflatten(Xr.dataXd, Xr.lengths)
    return data, backprop
