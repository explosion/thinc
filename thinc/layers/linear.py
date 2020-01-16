from typing import Tuple, Callable, Optional

from ..model import Model, CreateInit
from ..config import registry
from ..types import Array2d
from ..initializers import xavier_uniform_init, zero_init


InT = Array2d
OutT = Array2d


@registry.layers("Linear.v0")
def Linear(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
) -> Model[InT, OutT]:
    """Multiply inputs by a weights matrix and adds a bias vector."""
    model: Model[InT, OutT] = Model(
        "linear",
        forward,
        init=CreateInit({"W": init_W, "b": init_b}).init,
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

    def backprop(dY: OutT) -> InT:
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        return model.ops.gemm(dY, W)

    return Y, backprop
