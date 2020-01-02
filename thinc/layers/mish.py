from typing import Tuple, Callable, Optional, TypeVar

from ..model import Model, create_init
from ..initializers import xavier_uniform_init, zero_init
from ..types import Array
from .chain import chain
from .layernorm import LayerNorm
from .dropout import Dropout


InputType = TypeVar("InputType", bound=Array)
OutputType = TypeVar("OutputType", bound=Array)


def Mish(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
    dropout: Optional[float],
    normalize: bool = False,
) -> Model:
    """Dense layer with mish activation.
    https://arxiv.org/pdf/1908.08681.pdf
    """
    model = Model(
        "mish",
        forward,
        init=create_init({"W": init_W, "b": init_b}),
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
    )
    if normalize is not None:
        model = chain(model, LayerNorm())
    if dropout is not None:
        model = chain(model, Dropout(dropout))
    if nO is not None and nI is not None:
        model.initialize()
    return model


def forward(model: Model, X: InputType, is_train: bool) -> Tuple[OutputType, Callable]:
    W = model.get_param("W")
    b = model.get_param("b")
    Y_pre_mish = model.ops.gemm(X, W, trans2=True)
    Y_pre_mish += b
    Y = model.ops.mish(Y_pre_mish)

    def backprop(dY: OutputType) -> InputType:
        dY_pre_mish = model.ops.backprop_mish(dY, Y_pre_mish)
        model.inc_grad("W", model.ops.gemm(dY_pre_mish, X, trans1=True))
        model.inc_grad("b", dY_pre_mish.sum(axis=0))
        dX = model.ops.gemm(dY_pre_mish, W)
        return dX

    return Y, backprop
