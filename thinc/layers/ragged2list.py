from typing import Tuple, Callable, TypeVar, cast

from ..model import Model
from ..config import registry
from ..types import Ragged, ListXd


InT = Ragged
OutT = TypeVar("OutT", bound=ListXd)


@registry.layers("ragged2list.v1")
def ragged2list() -> Model[InT, OutT]:
    """Transform sequences from a ragged format into lists."""
    return Model("ragged2list", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    lengths = Xr.lengths

    def backprop(dXs: OutT) -> InT:
        return Ragged(model.ops.flatten(dXs, pad=0), lengths)

    data = cast(OutT, model.ops.unflatten(Xr.dataXd, Xr.lengths))
    return data, backprop
