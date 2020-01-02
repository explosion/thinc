from typing import Tuple, Callable, TypeVar

from ..model import Model
from ..types import Array


InputType = TypeVar("InputType", bound=Array)
OutputType = TypeVar("OutputType", bound=Array)


def ExtractWindow(window_size: int = 1) -> Model:
    """For each vector in an input, construct an output vector that contains the
    input and a window of surrounding vectors. This is one step in a convolution.
    """
    return Model("extract_window", forward, attrs={"window_size": window_size})


def forward(model: Model, X: InputType, is_train: bool) -> Tuple[OutputType, Callable]:
    nW = model.get_attr("window_size")
    Y = model.ops.seq2col(X, nW)

    def backprop(dY: OutputType) -> InputType:
        return model.ops.backprop_seq2col(dY, nW)

    return Y, backprop
