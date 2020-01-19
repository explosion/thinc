from typing import Optional, Tuple, Callable

from ..types import Array2d
from ..model import Model
from ..config import registry
from ..util import get_width


InT = Array2d
OutT = Array2d


@registry.layers("MultiSoftmax.v0")
def MultiSoftmax(nOs: Tuple[int, ...], nI: Optional[int] = None) -> Model[InT, OutT]:
    """Neural network layer that predicts several multi-class attributes at once.
    For instance, we might predict one class with 6 variables, and another with 5.
    We predict the 11 neurons required for this, and then softmax them such
    that columns 0-6 make a probability distribution and columns 6-11 make another.
    """
    return Model(
        "multisoftmax",
        forward,
        init=init,
        dims={"nO": sum(nOs), "nI": nI},
        attrs={"nOs": nOs},
        params={"W": None, "b": None},
    )


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    nOs = model.get_attr("nOs")
    W = model.get_param("W")
    b = model.get_param("b")

    def backprop(dY: OutT) -> InT:
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        model.inc_grad("b", dY.sum(axis=0))
        return model.ops.gemm(dY, W)

    Y = model.ops.gemm(X, W)
    Y += b
    i = 0
    for out_size in nOs:
        model.ops.softmax(Y[:, i : i + out_size], inplace=True)
        i += out_size
    return Y, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> Model[InT, OutT]:
    if X is not None:
        model.set_dim("nI", get_width(X))
    nO = model.get_dim("nO")
    nI = model.get_dim("nI")
    model.set_param("W", model.ops.alloc_f2d(nO, nI))
    model.set_param("b", model.ops.alloc_f1d(nO))
    return model
