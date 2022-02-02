from typing import Tuple, Callable, Optional, cast

from ..model import Model
from ..config import registry
from ..types import Floats2d, Floats1d
from ..initializers import zero_init
from ..util import get_width, partial


InT = Floats2d
OutT = Floats2d


@registry.layers("Softmax.v1")
def Softmax(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    init_W: Callable = zero_init,
    init_b: Callable = zero_init,
    normalize: bool = True,
) -> Model[InT, OutT]:
    return Model(
        "softmax",
        forward,
        init=partial(init, init_W, init_b),
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
        attrs={"normalize_softmax": normalize},
    )


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    normalized = model.attrs["normalize_softmax"] or is_train
    W = cast(Floats2d, model.get_param("W"))
    b = cast(Floats1d, model.get_param("b"))
    Y = model.ops.affine(X, W, b)
    if normalized:
        Y = model.ops.softmax(Y)

    def backprop(dY: InT) -> OutT:
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        return model.ops.gemm(dY, W)

    def backprop_unnormalized(dY: InT):
        msg = "backprop is not supported for an unnormalized Softmax layer"
        raise ValueError(msg)

    if normalized:
        return Y, backprop
    else:
        return Y, backprop_unnormalized


def init(
    init_W: Callable,
    init_b: Callable,
    model: Model[InT, OutT],
    X: Optional[InT] = None,
    Y: Optional[OutT] = None,
) -> Model[InT, OutT]:
    if X is not None and model.has_dim("nI") is None:
        model.set_dim("nI", get_width(X))
    if Y is not None and model.has_dim("nO") is None:
        model.set_dim("nO", get_width(Y))
    model.set_param("W", init_W(model.ops, (model.get_dim("nO"), model.get_dim("nI"))))
    model.set_param("b", init_b(model.ops, (model.get_dim("nO"),)))
    return model
