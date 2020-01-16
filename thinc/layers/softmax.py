from typing import Tuple, Callable, Optional

from ..model import Model, create_init
from ..config import registry
from ..types import Array2d
from ..initializers import zero_init


InT = Array2d
OutT = Array2d


@registry.layers("Softmax.v0")
def Softmax(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    init_W: Callable = zero_init,
    init_b: Callable = zero_init
) -> Model[InT, OutT]:
    model: Model[InT, OutT] = Model(
        "softmax",
        forward,
        init=create_init({"W": init_W, "b": init_b}).init,
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
    )
    if nO is not None and nI is not None:
        model.initialize()
    return model


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    W = model.get_param("W")
    b = model.get_param("b")

    Y = model.ops.gemm(X, W, trans2=True)
    Y += b
    model.ops.softmax(Y, inplace=True)

    def backprop(dY: InT) -> OutT:
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        return model.ops.gemm(dY, W)

    return Y, backprop
