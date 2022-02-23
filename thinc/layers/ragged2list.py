from typing import Tuple, Callable, List, cast

from ..model import Model
from ..config import registry
from ..types import Ragged, Array2d, ArrayXd_Iterable, ArrayXd_Concatenable


InT = Ragged
OutT = List[ArrayXd_Concatenable]


@registry.layers("ragged2list.v1")
def ragged2list() -> Model[InT, OutT]:
    """Transform sequences from a ragged format into lists."""
    return Model("ragged2list", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    lengths = Xr.lengths

    def backprop(dXs: OutT) -> InT:
        return Ragged(model.ops.flatten(cast(List[ArrayXd_Concatenable], dXs), pad=0), lengths)

    data: List[ArrayXd_Concatenable] = model.ops.unflatten(cast(ArrayXd_Iterable, Xr.dataXd), Xr.lengths)
    return data, backprop
