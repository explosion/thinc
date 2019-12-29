from typing import Tuple, Callable, Optional

from .base import Model, create_init
from .dropout import Dropout
from .layernorm import LayerNorm
from .chain import chain
from ..initializers import xavier_uniform_init, zero_init
from ..types import Array


def Maxout(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    nP: int = 3,
    *,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
    dropout: Optional[float],
    normalize: bool = False,
) -> Model:
    model = Model(
        "maxout",
        forward,
        init=create_init({"W": init_W, "b": init_b}),
        dims={"nO": nO, "nI": nI, "nP": nP},
        params={"W": None, "b": None},
        layers=[],
        attrs={},
    )
    if normalize is not None:
        model = chain(model, LayerNorm())
    if dropout is not None:
        model = chain(model, Dropout(dropout))
    if nO is not None and nI is not None:
        model.initialize()
    return model


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    nO = model.get_dim("nO")
    nP = model.get_dim("nP")
    nI = model.get_dim("nI")
    b = model.get_param("b")
    W = model.get_param("W")
    W = W.reshape((nO * nP, nI))
    Y = model.ops.gemm(X, W, trans2=True)
    Y += b.reshape((nO * nP,))
    Y = Y.reshape((Y.shape[0], nO, nP))
    best, which = model.ops.maxout(Y)

    def finish_update(d_best: Array):
        dY = model.ops.backprop_maxout(d_best, which, nP)
        dY = dY.reshape((dY.shape[0], nO * nP))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True).reshape((nO, nP, nI)))
        model.inc_grad("b", dY.sum(axis=0))
        return model.ops.gemm(dY, W.reshape((nO * nP, nI)))

    return best, finish_update
