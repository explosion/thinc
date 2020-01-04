from typing import Optional
from ..model import Model


def bidirectional(l2r: Model, r2l: Optional[Model] = None) -> Model:
    """Stitch two RNN models into a bidirectional layer."""
    if r2l is None:
        r2l = l2r.copy()
    return Model(f"bi{l2r.name}", forward, layers=[l2r, r2l])


def forward(model, Xs, is_train):
    l2r, r2l = model.layers
    nO = model.get_dim("nO")
    reverse_Xs = [l2r.ops.xp.ascontiguousarray(X[::-1]) for X in Xs]
    l2r_Zs, bp_l2r_Zs = l2r(Xs, is_train)
    r2l_Zs, bp_r2l_Zs = r2l(reverse_Xs, is_train)

    def backprop(dZs, sgd=None):
        d_l2r_Zs = []
        d_r2l_Zs = []
        for dZ in dZs:
            l2r_fwd = dZ[:, :nO]
            r2l_fwd = dZ[:, nO:]
            d_l2r_Zs.append(l2r.ops.xp.ascontiguousarray(l2r_fwd))
            d_r2l_Zs.append(l2r.ops.xp.ascontiguousarray(r2l_fwd[::-1]))
        dXs_l2r = bp_l2r_Zs(d_l2r_Zs)
        dXs_r2l = bp_r2l_Zs(d_r2l_Zs)
        return [dXf + dXb[::-1] for dXf, dXb in zip(dXs_l2r, dXs_r2l)]

    Zs = [l2r.ops.xp.hstack((Zf, Zb[::-1])) for Zf, Zb in zip(l2r_Zs, r2l_Zs)]
    return Zs, backprop
