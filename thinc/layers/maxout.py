from typing import Tuple, Callable, Optional, Dict, cast

from ..model import Model
from ..config import registry
from ..initializers import xavier_uniform_init, zero_init
from ..types import Array2d
from ..util import get_width
from .dropout import Dropout
from .layernorm import LayerNorm
from .chain_module import chain


InT = Array2d
OutT = Array2d


@registry.layers("Maxout.v0")
def Maxout(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    nP: Optional[int] = 3,
    *,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
    dropout: Optional[float] = None,
    normalize: bool = False,
) -> Model[InT, OutT]:
    model: Model[InT, OutT] = Model(
        "maxout",
        forward,
        init=create_init({"W": init_W, "b": init_b}),
        dims={"nO": nO, "nI": nI, "nP": nP},
        params={"W": None, "b": None},
    )
    if normalize:
        model = chain(model, LayerNorm(nI=nO))
    if dropout is not None:
        model = chain(model, cast(Model[InT, OutT], Dropout(dropout)))
    if nO is not None and nI is not None and nP is not None:
        model.initialize()
    return model


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    nO = model.get_dim("nO")
    nP = model.get_dim("nP")
    nI = model.get_dim("nI")
    b = model.get_param("b")
    W = model.get_param("W")
    W = W.reshape((nO * nP, nI))
    Y = model.ops.gemm(X, W, trans2=True)
    Y += b.reshape((nO * nP,))
    Y = Y.reshape((Y.shape[0], nO, nP))
    best, which = model.ops.maxout(Y)

    def backprop(d_best: OutT) -> InT:
        dY = model.ops.backprop_maxout(d_best, which, nP)
        model.inc_grad("b", dY.sum(axis=0))
        dY = dY.reshape((dY.shape[0], nO * nP))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True).reshape((nO, nP, nI)))
        return model.ops.gemm(dY, W.reshape((nO * nP, nI)))

    return best, backprop


class create_init:
    """Create an init function, given a dictionary of parameter initializers."""

    def __init__(self, initializers: Dict[str, Callable]):
        self.initializers = initializers

    def __call__(self, model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
    ) -> None:
        if X is not None:
            model.set_dim("nI", get_width(X))
        if Y is not None:
            model.set_dim("nO", get_width(Y))
        W = model.ops.alloc_f3d(
            model.get_dim("nO"), model.get_dim("nP"), model.get_dim("nI")
        )
        b = model.ops.alloc_f2d(model.get_dim("nO"), model.get_dim("nP"))
        if "W" in self.initializers:
            self.initializers["W"](W, inplace=True)
        if "b" in self.initializers:
            self.initializers["b"](b, inplace=True)
        model.set_param("W", W)
        model.set_param("b", b)
