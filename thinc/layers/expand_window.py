from typing import Tuple, Callable

from ..model import Model
from ..config import registry
from ..types import Floats2d


InT = Floats2d
OutT = Floats2d


@registry.layers("expand_window.v1")
def expand_window(window_size: int = 1) -> Model[InT, OutT]:
    """For each vector in an input, construct an output vector that contains the
    input and a window of surrounding vectors. This is one step in a convolution.
    """
    return Model("expand_window", forward, attrs={"window_size": window_size})


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    nW = model.attrs["window_size"]
    if len(X) > 0:
        Y = model.ops.seq2col(X, nW)

    def backprop(dY: OutT) -> InT:
        return model.ops.backprop_seq2col(dY, nW)

    if len(X) == 0:
        return X, backprop
    return Y, backprop
