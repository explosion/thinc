from typing import Optional, Tuple
from ..backends import Ops
from ..model import Model
from ..types import Array, Padded


def bidirectional(l2r: Model, r2l: Optional[Model] = None) -> Model:
    """Stitch two RNN models into a bidirectional layer. Expects squared sequences."""
    if r2l is None:
        r2l = l2r.copy()
    return Model(f"bi{l2r.name}", forward, layers=[l2r, r2l])


def forward(model: Model, X: Padded, is_train: bool):
    l2r, r2l = model.layers

    X_rev = _reverse(model.ops, X)
    l2r_Z, bp_l2r_Z = l2r(X, is_train)
    r2l_Z, bp_r2l_Z = r2l(X_rev, is_train)
    Z = _concatenate(model.ops, l2r_Z, r2l_Z)

    def backprop(dZ, sgd=None):
        d_l2r_Z, d_r2l_Z = _split(model.ops, dZ)
        dXs_l2r = bp_l2r_Zs(d_l2r_Z)
        dXs_r2l = bp_r2l_Zs(d_r2l_Z)
        return _sum(dX_l2r, dX_r2l)

    return Z, backprop


def _reverse(ops, Xp: Padded) -> Padded:
    reverse_X = Xp.data[::-1]
    return Padded(reverse_X, Xp.size_at_t)


def _concatenate(ops, l2r: Padded, r2l: Padded) -> Padded:
    concatenated = ops.xp.hstack((l2r.data, r2l.data), axis=-1)
    return Padded(concatenated, l2r.size_at_t)


def _split(ops: Ops, Xp: Padded) -> Tuple[Padded, Padded]:
    half = Xp.data.shape[-1] // 2
    X_l2r = Xp.data[..., :half]
    X_r2l = Xp.data[..., half:]
    return (Padded(X_l2r, Xp.size_at_t), Padded(X_r2l, Xp.size_at_t))


def _sum(ops: Ops, Xp: Padded, Yp: Padded) -> Padded:
    return Padded(Xp.data + Yp.data, Xp.size_at_t)
