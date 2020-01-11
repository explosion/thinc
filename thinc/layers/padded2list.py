from typing import Tuple, Callable, List

from ..types import Padded, Array2d
from ..model import Model
from ..config import registry


@registry.layers("padded2list.v0")
def padded2list() -> Model[Padded, List[Array2d]]:
    """Create a layer to convert a Padded input into a list of arrays."""
    return Model(f"padded2list", forward)


def forward(
    model: Model[Padded, List[Array2d]], Xp: Padded, is_train: bool
) -> Tuple[List[Array2d], Callable]:

    Ys = model.ops.padded2list(Xp)

    def backprop(dYs: List[Array2d]) -> Padded:
        return model.ops.list2padded(dYs)

    return Ys, backprop
