from typing import Tuple, Callable, TypeVar, cast

from ..types import Padded, List2d
from ..model import Model
from ..config import registry


InT = TypeVar("InT", bound=List2d)
OutT = Padded


@registry.layers("list2padded.v1")
def list2padded() -> Model[InT, OutT]:
    """Create a layer to convert a list of array inputs into Padded."""
    return Model(f"list2padded", forward)


def forward(model: Model[InT, OutT], Xs: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Yp = model.ops.list2padded(Xs)

    def backprop(dYp: OutT) -> InT:
        return cast(InT, model.ops.padded2list(dYp))

    return Yp, backprop
