from typing import Callable, Tuple, cast, TypeVar

from ..model import Model
from ..config import registry
from ..types import Ragged, ArrayXd

OutT = TypeVar("OutT", bound=ArrayXd)
OutT_co = TypeVar("OutT_co", bound=ArrayXd, covariant=True)


@registry.layers("reduce_first.v1")
def reduce_first() -> Model[Ragged, OutT_co]:
    """Reduce ragged-formatted sequences to their first element."""
    return Model("reduce_first", forward)


def forward(
    model: Model[Ragged, OutT], Xr: Ragged, is_train: bool
) -> Tuple[OutT, Callable]:
    starts = model.ops.alloc1i(Xr.lengths.shape[0])
    starts[1:] += Xr.lengths.cumsum()[:-1]
    X = cast(OutT, Xr.dataXd)
    Y = cast(OutT, X[starts])
    x_shape = Xr.dataXd.shape
    lengths = Xr.lengths

    def backprop(dY: OutT) -> Ragged:
        dX = cast(OutT, model.ops.alloc(x_shape, dtype=dY.dtype))
        dX[starts] = dY  # type: ignore[assignment]
        return Ragged(dX, lengths)

    return Y, backprop
