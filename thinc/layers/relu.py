from typing import Tuple, Callable, Optional, TypeVar
from typing_extensions import Literal

from ..model import Model, create_init
from ..initializers import xavier_uniform_init, zero_init
from ..types import Array, Floats1d, Floats2d
from .chain import chain
from .layernorm import LayerNorm
from .dropout import Dropout


B = TypeVar("B", bound=int)
O = TypeVar("O", bound=int)
I = TypeVar("I", bound=int)


def ReLu(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
    dropout: Optional[float] = None,
    normalize: bool = False,
) -> Model:
    model = Model[Floats2d[B, O], Floats2d[B, O]](
        "relu",
        forward,
        init=create_init({"W": init_W, "b": init_b}),
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
    )
    if normalize:
        model = chain(model, LayerNorm())
    if dropout is not None:
        model = chain(model, Dropout(dropout))
    if nO is not None and nI is not None:
        model.initialize()
    return model


def forward(
        model: Model,
        X: Floats2d[B, I],
        is_train: bool
) -> Tuple[Floats2d[B, O], Callable]:
    W: Floats2d[O, I] = model.get_param("W")
    b: Floats1d[O] = model.get_param("b")
    Y: Floats2d[B, O] = model.ops.gemm(X, W, trans2=True)
    Y += b
    model.ops.relu(Y, inplace=True)

    def backprop(dY: Floats2d[B, O]) -> Floats2d[B, I]:
        dY = model.ops.backprop_relu(dY, Y)
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        return model.ops.gemm(dY, W)

    return Y, backprop
