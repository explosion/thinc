from typing import Tuple, Callable, cast, List

from ..types import Padded, Array2d
from ..model import Model
from ..config import registry


InT = Padded
OutT = List[Array2d]


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
