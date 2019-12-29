from typing import Tuple, Callable, Optional

from .base import Model, Array
from ..initializers import xavier_uniform_init, zero_init
from ..util import get_width


def Mish(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
) -> Model:
    """Dense layer with mish activation.
    https://arxiv.org/pdf/1908.08681.pdf
    """
    model = Model(
        "mish",
        forward,
        init=create_init(init_W, init_b),
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
        layers=[],
        attrs={},
    )
    if nO is not None and nI is not None:
        model.initialize()
    return model


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    W = model.get_attr("W")
    b = model.get_attr("b")
    Y1 = model.ops.affine(W, b, X)
    Y2 = model.ops.mish(Y1)

    def mish_backward(dY2: Array) -> Array:
        dY1 = model.ops.backprop_mish(dY2, Y1)
        model.inc_grad("W", model.ops.gemm(dY1, X, trans1=True))
        model.inc_grad("b", dY1.sum(axis=0))
        dX = model.ops.gemm(dY1, W)
        return dX

    return Y2, mish_backward


def create_init(init_W: Callable, init_b: Callable) -> Callable:
    def do_mish_init(
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

    return do_mish_init
