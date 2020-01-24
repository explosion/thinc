from typing import Tuple, Callable, Optional, cast

from ..model import Model
from ..config import registry
from ..initializers import glorot_uniform_init, zero_init
from ..types import Array2d, Array3d
from ..util import get_width, partial
from .dropout import Dropout
from .layernorm import LayerNorm
from .chain import chain


InT = Array2d
OutT = Array2d


@registry.layers("Maxout.v0")
def Maxout(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    nP: Optional[int] = 3,
    *,
    init_W: Callable = glorot_uniform_init,
    init_b: Callable = zero_init,
    dropout: Optional[float] = None,
    normalize: bool = False,
) -> Model[InT, OutT]:
    model: Model[InT, OutT] = Model(
        "maxout",
        forward,
        init=partial(init, init_W, init_b),
        dims={"nO": nO, "nI": nI, "nP": nP},
        params={"W": None, "b": None},
    )
    if normalize:
        model = chain(model, LayerNorm(nI=nO))
    if dropout is not None:
        model = chain(model, cast(Model[InT, OutT], Dropout(dropout)))
    return model


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    nO = model.get_dim("nO")
    nP = model.get_dim("nP")
    nI = model.get_dim("nI")
    b = model.get_param("b")
    W = model.get_param("W")
    W = cast(Array2d, W.reshape((nO * nP, nI)))
    Y = model.ops.gemm(X, W, trans2=True)
    Y += b.reshape((nO * nP,))
    Z = cast(Array3d, Y.reshape((Y.shape[0], nO, nP)))
    best, which = model.ops.maxout(Z)

    def backprop(d_best: OutT) -> InT:
        dZ = model.ops.backprop_maxout(d_best, which, nP)
        model.inc_grad("b", dZ.sum(axis=0))
        dY = dZ.reshape((dZ.shape[0], nO * nP))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True).reshape((nO, nP, nI)))
        return model.ops.gemm(dY, W.reshape((nO * nP, nI)))

    return best, backprop


def init(
    init_W: Callable,
    init_b: Callable,
    model: Model[InT, OutT],
    X: Optional[InT] = None,
    Y: Optional[OutT] = None,
) -> Model[InT, OutT]:
    if X is not None:
        model.set_dim("nI", get_width(X))
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    W_shape = (model.get_dim("nO"), model.get_dim("nP"), model.get_dim("nI"))
    model.set_param("W", init_W(model.ops, W_shape))
    model.set_param("b", init_b(model.ops, (model.get_dim("nO"), model.get_dim("nP"))))
    return model
