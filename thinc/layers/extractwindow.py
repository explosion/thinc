from typing import Tuple, Callable

from ..model import Model
from ..types import Array


def ExtractWindow(window_size: int = 1) -> Model:
    return Model(
        "extract_window",
        forward,
        attrs={"window_size": window_size},
        init=None,
        dims={},
        params={},
        layers=[],
    )


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    nW = model.get_attr("window_size")
    Y = model.ops.seq2col(X, nW)

    def backprop_convolution(dY: Array) -> Array:
        return model.ops.backprop_seq2col(dY, nW)

    return Y, backprop_convolution
