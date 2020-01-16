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


@registry.layers("Mish.v0")
def Mish(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
    dropout: Optional[float] = None,
    normalize: bool = False,
) -> Model[InT, OutT]:
    """Dense layer with mish activation.
    https://arxiv.org/pdf/1908.08681.pdf
    """
    model: Model[InT, OutT] = Model(
        "mish",
        forward,
        init=create_init({"W": init_W, "b": init_b}),
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
    )
    if normalize:
        model = chain(model, cast(Model[InT, OutT], LayerNorm(nI=nO)))
    if dropout is not None:
        model = chain(model, cast(Model[InT, OutT], Dropout(dropout)))
    if nO is not None and nI is not None:
        model.initialize()
    return model


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    W = model.get_param("W")
    b = model.get_param("b")
    Y_pre_mish = model.ops.gemm(X, W, trans2=True)
    Y_pre_mish += b
    Y = model.ops.mish(Y_pre_mish)

    def backprop(dY: OutT) -> InT:
        dY_pre_mish = model.ops.backprop_mish(dY, Y_pre_mish)
        model.inc_grad("W", model.ops.gemm(dY_pre_mish, X, trans1=True))
        model.inc_grad("b", dY_pre_mish.sum(axis=0))
        dX = model.ops.gemm(dY_pre_mish, W)
        return dX

    return Y, backprop
