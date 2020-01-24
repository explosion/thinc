from typing import Tuple, Callable, cast

from ..model import Model
from ..config import registry
from ..types import Array2d


InT = Array2d
OutT = Array2d


@registry.layers("expand_window.v0")
def expand_window(window_size: int = 1) -> Model[InT, OutT]:
    """For each vector in an input, construct an output vector that contains the
    input and a window of surrounding vectors. This is one step in a convolution.
    """
    return Model("expand_window", forward, attrs={"window_size": window_size})


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    nW = model.attrs["window_size"]
    Y = model.ops.seq2col(X, nW)

    def backprop(dY: OutT) -> InT:
        return cast(Array2d, model.ops.backprop_seq2col(dY, nW))

    return Y, backprop
