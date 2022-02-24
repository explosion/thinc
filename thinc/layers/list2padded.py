from typing import Tuple, Callable, TypeVar, List, Union

from ..types import Padded, List2d, Array2d
from ..model import Model
from ..config import registry


InT = List2d
InT_co = TypeVar("InT_co", bound=Union[List2d, List[Array2d]], covariant=True)
OutT = Padded


@registry.layers("list2padded.v1")
def list2padded() -> Model[InT_co, OutT]:
    """Create a layer to convert a list of array inputs into Padded."""
    return Model(f"list2padded", forward)


def forward(model: Model[InT, OutT], Xs: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Yp = model.ops.list2padded(Xs)  # type: ignore

    def backprop(dYp: OutT) -> InT:
        return model.ops.padded2list(dYp)  # type: ignore

    return Yp, backprop
