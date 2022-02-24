from typing import Tuple, Callable, cast, List, TypeVar, Union

from ..types import Padded, Array2d
from ..model import Model
from ..config import registry


InT = Padded
OutT_member_co = TypeVar("OutT_member_co", bound=Array2d, covariant=True)
OutT = List[OutT_member_co]


@registry.layers("padded2list.v1")
def padded2list() -> Model[InT, OutT]:
    """Create a layer to convert a Padded input into a list of arrays."""
    return Model(f"padded2list", forward)


def forward(model: Model[InT, OutT], Xp: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Ys = model.ops.padded2list(Xp)

    def backprop(dYs: OutT) -> InT:
        dYp = model.ops.list2padded(dYs)
        assert isinstance(dYp, Padded)
        return dYp

    return cast(OutT, Ys), backprop
