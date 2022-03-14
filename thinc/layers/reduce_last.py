from typing import Callable, Tuple, cast, TypeVar

from ..model import Model
from ..config import registry
from ..types import Ragged, ArrayXd
from ..util import ArrayInfo

OutT = TypeVar("OutT", bound=ArrayXd)
OutT_co = TypeVar("OutT_co", bound=ArrayXd, covariant=True)


@registry.layers("reduce_last.v1")
def reduce_last() -> Model[Ragged, OutT_co]:
    """Reduce ragged-formatted sequences to their last element."""
    return Model("reduce_last", forward)


def forward(
    model: Model[Ragged, OutT_co], Xr: Ragged, is_train: bool
) -> Tuple[OutT_co, Callable[[OutT], Ragged]]:
    ends = Xr.lengths.cumsum() - 1
    Y = cast(OutT_co, Xr.dataXd[ends])
    x_shape = Xr.dataXd.shape
    lengths = Xr.lengths
    array_info = ArrayInfo.from_array(Y)

    def backprop(dY: OutT) -> Ragged:
        array_info.check_consistency(dY)
        dX: OutT = model.ops.alloc(x_shape, dtype=dY.dtype)
        dX[ends] = dY  # type: ignore[assignment]
        return Ragged(dX, lengths)

    return Y, backprop
