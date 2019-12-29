from typing import Tuple, Callable, Optional

from .base import Model, create_init
from .chain import chain
from .layernorm import LayerNorm
from .dropout import Dropout
from ..initializers import xavier_uniform_init, zero_init
from ..types import Array


def ReLu(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
    dropout: Optional[float] = None,
    normalize: bool = False,
) -> Model:
    model = Model(
        "relu",
        forward,
        init=create_init({"W": init_W, "b": init_b}),
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
        attrs={},
        layers=[],
    )
    if normalize:
        model = chain(model, LayerNorm())
    if dropout is not None:
        model = chain(model, Dropout(dropout))
    if nO is not None and nI is not None:
        model.initialize()
    return model


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    W = model.get_param("W")
    b = model.get_param("b")

    Y = model.ops.gemm(X, W, trans2=True)
    Y += b
    model.ops.relu(Y, inplace=True)

    def relu_backward(dY: Array) -> Array:
        dY = model.ops.backprop_relu(dY, Y)
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        return model.ops.gemm(dY, W)

    return Y, relu_backward
