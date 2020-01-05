from typing import Tuple, Callable, Optional

from ..model import Model
from ..types import Array
from ..backends import Ops
from ..util import get_width


# TODO: Make more specific
InT = Array
OutT = Array


def LayerNorm(nO: Optional[int] = None) -> Model[InT, OutT]:
    return Model(
        "layernorm",
        forward,
        init=init,
        dims={"nO": nO, "nI": nO},
        params={"G": None, "b": None},
    )


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    N, mu, var = _get_moments(model.ops, X)
    Xhat = (X - mu) * var ** (-1.0 / 2.0)
    Y, backprop_rescale = _begin_update_scale_shift(model, Xhat)

    def backprop(dY: OutT) -> InT:
        dY = backprop_rescale(dY)
        dist, sum_dy, sum_dy_dist = _get_d_moments(model.ops, dY, X, mu)
        d_xhat = N * dY - sum_dy - dist * var ** (-1.0) * sum_dy_dist
        d_xhat *= var ** (-1.0 / 2)
        d_xhat /= N
        return d_xhat

    return Y, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> None:
    if X is not None:
        X_width = get_width(X)
        model.set_dim("nI", X_width)
        model.set_dim("nO", X_width)
    if Y is not None:
        Y_width = get_width(Y)
        model.set_dim("nI", Y_width)
        model.set_dim("nO", Y_width)
    nO = model.get_dim("nO")
    model.set_param("G", model.ops.allocate((nO,)))
    model.set_param("b", model.ops.allocate((nO,)))


def _begin_update_scale_shift(model: Model[InT, OutT], X: InT) -> Tuple[InT, Callable]:
    G = model.get_param("G")
    b = model.get_param("b")
    Y = X * G
    Y += b

    def finish_update_scale_shift(dY: OutT) -> InT:
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("G", (dY * X).sum(axis=0))
        return dY * G

    return Y, finish_update_scale_shift


def _get_moments(ops: Ops, X: Array) -> Tuple[Array, Array, Array]:
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True) + 1e-08
    return ops.asarray([X.shape[1]], dtype="f"), mu, var


def _get_d_moments(
    ops: Ops, dy: Array, X: Array, mu: Array
) -> Tuple[Array, Array, Array]:
    dist = X - mu
    return (
        dist,
        ops.xp.sum(dy, axis=1, keepdims=True),
        ops.xp.sum(dy * dist, axis=1, keepdims=True),
    )
