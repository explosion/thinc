from typing import Optional, Tuple
from ..model import Model
from ..types import Array


def bidirectional(l2r: Model, r2l: Optional[Model] = None) -> Model:
    """Stitch two RNN models into a bidirectional layer. Expects squared sequences."""
    if r2l is None:
        r2l = l2r.copy()
    return Model(f"bi{l2r.name}", forward, layers=[l2r, r2l])


def forward(model: Model, Xs: Tuple[Array, Array], is_train: bool):
    l2r, r2l = model.layers
    nO = model.get_dim("nO")
    
    Xs_rev = _reverse(model.ops, Xs)
    l2r_Zs, bp_l2r_Zs = l2r(Xs, is_train)
    r2l_Zs, bp_r2l_Zs = r2l(Xs_rev, is_train)
    Zs, split = _concatenate(model.ops, l2r_Zs, r2l_Zs)

    def backprop(dZs, sgd=None):
        d_l2r_Zs, d_r2l_Zs = split(dZs)
        dXs_l2r = bp_l2r_Zs(d_l2r_Zs)
        dXs_r2l = bp_r2l_Zs(d_r2l_Zs)
        return _sum(dXs_l2r, dXs_r2l)

    return Zs, backprop

# The padded format with the auxiliary array is a bit quirky, so isolate
# the operations on it so that the code above doesn't have to think about it.

def _reverse(ops, X_size_at_t):
    X, size_at_t = X_size_at_t
    reverse_X = X[::-1]
    reverse_size_at_t = ops.xp.ascontiguousarray(size_at_t[::-1])
    return reverse_X, reverse_size_at_t


def _concatenate(ops, l2r, r2l):
    size_at_t = l2r[1]
    concatenated = ops.xp.hstack((l2r[0], r2l[0]), axis=-1)
    return (concatenated, size_at_t)


def _split(ops, X_size_at_t):
    X, size_at_t = X_size_at_t
    half = X.shape[-1] // 2
    X_l2r = X[..., :half]
    X_r2l = X[..., half:]
    return ((X_l2r, size_at_t), (X_r2l, size_at_t))


def _sum(ops, X_size_at_t, Y_size_at_t):
    X, size_at_t = X_size_at_t
    Y, size_at_t = Y_size_at_t
    return (X+Y, size_at_t)
