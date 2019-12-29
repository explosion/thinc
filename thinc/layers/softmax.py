from typing import Tuple, Callable, Optional

from .base import Model, Array, create_init
from .chain import chain
from .layernorm import LayerNorm
from .dropout import Dropout
from .initializers import xavier_uniform_init, zero_init


def Softmax(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    init_W: Callable = zero_init,
    init_b: Callable = zero_init
) -> Model:
    model = Model(
        "softmax",
        forward,
        init=create_init({"W": init_W, "b": init_b}),
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
        attrs={},
        layers=[],
    )
    if nO is not None and nI is not None:
        model.initialize()
    return model


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    W = model.get_param("W")
    b = model.get_param("b")

    Y = model.ops.gemm(X, W, trans2=True)
    Y += b
    model.ops.softmax(Y, inplace=True)

    def softmax_backward(dY: Array) -> Array:
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        return model.ops.gemm(dY, W)

    return Y, softmax_backward
