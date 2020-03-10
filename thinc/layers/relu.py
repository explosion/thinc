from typing import Tuple, Callable, Optional, cast, Dict, Union

from ..model import Model
from ..initializers import glorot_uniform_init, zero_init
from ..config import registry
from ..types import Floats2d, Floats1d
from ..util import get_width, partial
from .chain import chain
from .layernorm import LayerNorm
from .dropout import Dropout


InT = Floats2d
OutT = Floats2d


@registry.layers("Relu.v1")
def Relu(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    init_W: Callable = glorot_uniform_init,
    init_b: Callable = zero_init,
    dropout: Optional[float] = None,
    alphaLeaky: Optional[float] = 0,
    normalize: bool = False,
) -> Model[InT, OutT]:
    attrs: Dict[str, Union[None, int, float]] = {}
    attrs["alphaLeaky"] = alphaLeaky
    model: Model[InT, OutT] = Model(
        "relu",
        forward,
        init=partial(init, init_W, init_b),
        attrs=attrs,
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
    )
    if normalize:
        model = chain(model, LayerNorm(nI=nO))
    if dropout is not None:
        model = chain(model, cast(Model[Floats2d, Floats2d], Dropout(dropout)))

    return model


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    W = cast(Floats2d, model.get_param("W"))
    b = cast(Floats1d, model.get_param("b"))
    Y = model.ops.affine(X, W, b)
    alphaLeaky: float = model.attrs.get("alphaLeaky")
    Y = model.ops.relu(Y,alphaLeaky=alphaLeaky)

    def backprop(dY: OutT) -> InT:
        dY = model.ops.backprop_relu(dY, Y, alphaLeaky=alphaLeaky)
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        return model.ops.gemm(dY, W)

    return Y, backprop


def init(
    init_W: Callable,
    init_b: Callable,
    model: Model[InT, OutT],
    X: Optional[InT] = None,
    Y: Optional[OutT] = None,
) -> Model[InT, OutT]:
    if X is not None:
        model.set_dim("nI", get_width(X))
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    model.set_param("W", init_W(model.ops, (model.get_dim("nO"), model.get_dim("nI"))))
    model.set_param("b", init_b(model.ops, (model.get_dim("nO"),)))
    return model
