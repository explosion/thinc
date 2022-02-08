from typing import Callable, Tuple, cast

from ..model import Model
from ..config import registry
from ..types import Floats2d, Ragged


InT = Ragged
OutT = Floats2d


@registry.layers("reduce_sum.v1")
def reduce_sum() -> Model[InT, OutT]:
    return Model("reduce_sum", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y = model.ops.reduce_sum(cast(Floats2d, Xr.data), Xr.lengths)
    y_shape = Y.shape
    lengths = Xr.lengths

    def backprop(dY: OutT) -> InT:
        if dY.shape != y_shape:
            raise ValueError(f"Shape mismatch in backprop. Y: {y_shape}, dY: {dY.shape}")
        return Ragged(model.ops.backprop_reduce_sum(dY, lengths), lengths)

    return Y, backprop
