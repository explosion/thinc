from typing import Tuple, Callable, List

from ..types import Padded, Array2d
from ..model import Model
from ..config import registry


@registry.layers("list2padded.v0")
def list2padded() -> Model[List[Array2d], Padded]:
    """Create a layer to convert a list of array inputs into Padded."""
    return Model(f"list2padded", forward)


def forward(
    model: Model[List[Array2d], Padded], Xs: List[Array2d], is_train: bool
) -> Tuple[Padded, Callable]:

    Yp = model.ops.list2padded(Xs)

    def backprop(dYp: Padded) -> List[Array2d]:
        return model.ops.padded2list(dYp)

    return Yp, backprop
