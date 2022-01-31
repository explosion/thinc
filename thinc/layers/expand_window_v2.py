from typing import Tuple, Callable, cast

from ..model import Model
from ..config import registry
from ..types import Floats2d, Ragged


InT = Ragged
OutT = Ragged


@registry.layers("expand_window.v2")
def expand_window_ragged(window_size: int = 1) -> Model[InT, OutT]:
    """For each vector in an input, construct an output vector that contains the
    input and a window of surrounding vectors. This is one step in a convolution.
    """
    return Model("expand_window", forward_ragged, attrs={"window_size": window_size})


def forward_ragged(
    model: Model[InT, OutT], Xr: InT, is_train: bool
) -> Tuple[OutT, Callable]:
    nW = model.attrs["window_size"]
    Y = model.ops.seq2col(cast(Floats2d, Xr.data), nW, lengths=Xr.lengths)

    def backprop(dY: OutT) -> InT:
        dX = model.ops.backprop_seq2col(cast(Floats2d, dY.data), nW, lengths=Xr.lengths)
        return Ragged(dX, Xr.lengths)

    return Ragged(Y, Xr.lengths), backprop
