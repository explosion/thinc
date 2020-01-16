from typing import Tuple, Callable, Optional, cast

from ..model import Model, create_init
from ..initializers import xavier_uniform_init, zero_init
from ..config import registry
from ..types import Array2d
from .chain_module import chain
from .layernorm import LayerNorm
from .dropout import Dropout


InT = Array2d
OutT = Array2d


@registry.layers("ReLu.v0")
def ReLu(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
    dropout: Optional[float] = None,
    normalize: bool = False,
) -> Model[InT, OutT]:
    model: Model[InT, OutT] = Model(
        "relu",
        forward,
        init=CreateInit({"W": init_W, "b": init_b}).init,
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
    )
    if normalize:
        model = chain(model, LayerNorm(nI=nO))
    if dropout is not None:
        model = chain(model, cast(Model[Array2d, Array2d], Dropout(dropout)))
    if nO is not None and nI is not None:
        model.initialize()
    return model


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    W = model.get_param("W")
    b = model.get_param("b")

    Y = model.ops.gemm(X, W, trans2=True)
    Y += b
    model.ops.relu(Y, inplace=True)

    def backprop(dY: OutT) -> InT:
        dY = model.ops.backprop_relu(dY, Y)
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        return model.ops.gemm(dY, W)

    return Y, backprop
