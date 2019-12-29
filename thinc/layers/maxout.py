from typing import Tuple, Callable, Optional

from .base import Model, Array
from ..initializers import xavier_uniform_init, zero_init
from ..util import get_width


def Maxout(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    nP: int = 3,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
) -> Model:
    model = Model(
        "maxout",
        forward,
        init=create_init(init_W, init_b),
        dims={"nO": nO, "nI": nI, "nP": nP},
        params={"W": None, "b": None},
        layers=[],
        attrs={},
    )
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
        dY = model.ops.backprop_maxout(d_best, which, model.nP)
        model.d_b += dY.sum(axis=0)
        dY = dY.reshape((dY.shape[0], nO * nP))
        dW = model.ops.gemm(dY, X, trans1=True)
        model.inc_grad("W", dW.reshape((nO, nP, nI)))
        # Bop,opi->Bi
        return model.ops.gemm(dY, W.reshape((nO * nP, nI)))

    return best, finish_update


def create_init(init_W: Callable, init_b: Callable) -> Callable:
    def do_maxout_init(
        model: Model, X: Optional[Array] = None, Y: Optional[Array] = None
    ) -> None:
        if X is not None:
            model.set_dim("nI", get_width(X))
        if Y is not None:
            model.set_dim("nO", get_width(Y))
        W = model.ops.allocate((model.get_dim("nO"), model.get_dim("nI")))
        b = model.ops.allocate((model.get_dim("nO"),))
        init_W(W, inplace=True)
        init_b(b, inplace=True)
        model.set_param("W", W)
        model.set_param("b", b)

    return do_maxout_init
