from typing import Optional, Tuple
from thinc.types import Floats2d
from thinc.model import Model
from thinc.util import get_width


def MultiSoftmax(out_sizes: Tuple[int, ...], nI: Optional[int]=None):
    """Neural network layer that predicts several multi-class attributes at once.
    For instance, we might predict one class with 6 variables, and another with 5.
    We predict the 11 neurons required for this, and then softmax them such
    that columns 0-6 make a probability distribution and coumns 6-11 make another.
    """
    return Model[Floats2d, Floats2d](
        "multisoftmax",
        forward,
        init=init,
        dims={"nO": sum(out_sizes), "nI": nI},
        attrs={"out_sizes": out_sizes},
        params={"W": None, "b": None}
    )


def init(model: Model[Floats2d, Floats2d], X=None, Y=None):
    if X is not None:
        model.set_dim("nI", get_width(X))
    nO = model.get_dim("nO")
    nI = model.get_dim("nI")
    model.set_param("W", model.ops.allocate((nO, nI)))
    model.set_param("b", model.ops.allocate((nO,)))


def forward(model: Model[Floats2d, Floats2d], X: Floats2d, is_train):
    out_sizes = model.get_attr("out_sizes")
    W = model.get_param("W")
    b = model.get_param("b")
    
    def backprop(dY: Floats2d) -> Floats2d:
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        model.inc_grad("b", dY.sum(axis=0))
        return model.ops.gemm(dY, W)

    Y = model.ops.gemm(X, W)
    Y += b
    i = 0
    for out_size in out_sizes:
        model.ops.softmax(Y[:, i : i + out_size], inplace=True)
        i += out_size

    return Y, backprop
