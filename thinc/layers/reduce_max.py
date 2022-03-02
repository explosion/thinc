from typing import Tuple, Callable, cast

from ..types import Floats2d, Ragged
from ..model import Model
from ..config import registry
from ..util import ArrayInfo


InT = Ragged
OutT = Floats2d


@registry.layers("reduce_max.v1")
def reduce_max() -> Model[InT, OutT]:
    return Model("reduce_max", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y, which = model.ops.reduce_max(cast(Floats2d, Xr.data), Xr.lengths)
    lengths = Xr.lengths
    array_info = ArrayInfo.from_array(Y)

    def backprop(dY: OutT) -> InT:
        array_info.check_consistency(dY)
        return Ragged(model.ops.backprop_reduce_max(dY, which, lengths), lengths)

    return Y, backprop
