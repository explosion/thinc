from typing import Tuple, Callable, TypeVar, List, Union, cast

from ..types import Padded, Array2d
from ..model import Model
from ..config import registry


InT_member_co = TypeVar("InT_member_co", bound=Array2d, covariant=True)
InT = List[InT_member_co]
OutT = Padded


@registry.layers("list2padded.v1")
def list2padded() -> Model[InT, OutT]:
    """Create a layer to convert a list of array inputs into Padded."""
    return Model(f"list2padded", forward)


def forward(model: Model[InT, OutT], Xs: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Yp = model.ops.list2padded(Xs)  # type: ignore[misc]

    def backprop(dYp: OutT) -> InT:
        return model.ops.padded2list(dYp)

    return Yp, backprop
