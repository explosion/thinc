from typing import Tuple, Callable, Optional

from ..model import Model
from ..config import registry
from ..types import Array2d
from ..backends import Ops
from ..util import get_width


InT = Array2d


@registry.layers("LayerNorm.v0")
def LayerNorm(nI: Optional[int] = None) -> Model[InT, InT]:
    return Model(
        "layernorm", forward, init=init, dims={"nI": nI}, params={"G": None, "b": None}
    )


def forward(model: Model[InT, InT], X: InT, is_train: bool) -> Tuple[InT, Callable]:
    N, mu, var = _get_moments(model.ops, X)
    Xhat = (X - mu) * var ** (-1.0 / 2.0)
    Y, backprop_rescale = _begin_update_scale_shift(model, Xhat)

    def backprop(dY: InT) -> InT:
        dY = backprop_rescale(dY)
        dist, sum_dy, sum_dy_dist = _get_d_moments(model.ops, dY, X, mu)
        d_xhat = N * dY - sum_dy - dist * var ** (-1.0) * sum_dy_dist
        d_xhat *= var ** (-1.0 / 2)
        d_xhat /= N
        return d_xhat

    return Y, backprop


def init(
    model: Model[InT, InT], X: Optional[InT] = None, Y: Optional[InT] = None
) -> Model[InT, InT]:
    if X is not None:
        X_width = get_width(X)
        model.set_dim("nI", X_width)
    if Y is not None:
        Y_width = get_width(Y)
        model.set_dim("nI", Y_width)
    nI = model.get_dim("nI")
    model.set_param("G", model.ops.alloc_f1d(nI) + 1)
    model.set_param("b", model.ops.alloc_f1d(nI))
    return model


def _begin_update_scale_shift(model: Model[InT, InT], X: InT) -> Tuple[InT, Callable]:
    G = model.get_param("G")
    b = model.get_param("b")
    Y = X * G
    Y += b

    def finish_update_scale_shift(dY: InT) -> InT:
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("G", (dY * X).sum(axis=0))
        return dY * G

    return Y, finish_update_scale_shift


def _get_moments(ops: Ops, X: Array2d) -> Tuple[Array2d, Array2d, Array2d]:
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True) + 1e-08
    return ops.asarray([X.shape[1]], dtype="f"), mu, var


def _get_d_moments(
    ops: Ops, dy: Array2d, X: Array2d, mu: Array2d
) -> Tuple[Array2d, Array2d, Array2d]:
    dist = X - mu
    return (
        dist,
        ops.xp.sum(dy, axis=1, keepdims=True),
        ops.xp.sum(dy * dist, axis=1, keepdims=True),
    )
