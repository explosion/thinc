def forward(model, X):
    N, mu, var = _get_moments(model.ops, X)

    Xhat = (X - mu) * var ** (-1.0 / 2.0)

    y, backprop_rescale = _begin_update_scale_shift(model, Xhat)

    def backprop_layer_norm(dY):
        dy = backprop_rescale(dY)
        dist, sum_dy, sum_dy_dist = _get_d_moments(model.ops, dY, X, mu)
        d_xhat = N * dY - sum_dy - dist * var ** (-1.0) * sum_dy_dist
        d_xhat *= var ** (-1.0 / 2)
        d_xhat /= N
        return d_xhat

    return y, finish_update


def init(model, X=None, Y=None):
    if X is not None:
        X_width = util.get_width(X)
        model.set_dim("nI", X_width)
        model.set_dim("nO", X_width)
    if Y is not None:
        Y_width = util.get_width(Y)
        model.set_dim("nI", Y_width)
        model.set_dim("nO", Y_width)
    nO = model.get_dim("nO")
    model.set_param("G", model.ops.allocate((nO,)))
    model.set_param("b", model.ops.allocate((nO,)))


def make_LayerNorm(nO=None):
    return Model(
        "layernorm",
        forward,
        init=init,
        dims={"nO": nO, "nI": nO},
        params={"G": None, "b": None},
        layers=[],
        attrs={}
    )

def _begin_update_scale_shift(model, X):
    G = model.get_param("G")
    b = model.get_param("b")
    Y = X * G
    X += b

    def finish_update_scale_shift(dY):
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("G", (dY * X).sum(axis=0))
        return dY * G

    return X * G + b, finish_update_scale_shift


def _get_moments(ops, X):
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True) + 1e-08
    return ops.asarray([X.shape[1]], dtype="f"), mu, var


def _get_d_moments(ops, dy, X, mu):
    dist = X - mu
    return (
        dist,
        ops.xp.sum(dy, axis=1, keepdims=True),
        ops.xp.sum(dy * dist, axis=1, keepdims=True),
    )
