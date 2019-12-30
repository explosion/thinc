from typing import Tuple, Callable, Optional

from ..model import Model, create_init
from ..types import Array
from ..initializers import xavier_uniform_init, zero_init


def Affine(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
) -> Model:
    model = Model(
        "affine",
        forward,
        init=create_init({"W": init_W, "b": init_b}),
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
    )
    if nO is not None and nI is not None:
        model.initialize()
    return model


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    W = model.get_param("W")
    b = model.get_param("b")
    Y = model.ops.gemm(X, W, trans2=True)
    Y += b

    def backprop(dY: Array) -> Array:
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        return model.ops.gemm(dY, W)

    return Y, backprop
