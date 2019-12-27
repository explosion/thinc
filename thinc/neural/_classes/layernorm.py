from ... import describe
from .model import Model
from .. import util


def _init_to_one(W, ops):
    W.fill(1.0)


@describe.attributes(
    G=describe.Weights("Scaling vector", lambda obj: (obj.nO,), _init_to_one),
    b=describe.Weights("Bias vector", lambda obj: (obj.nO,)),
    d_G=describe.Gradient("G"),
    d_b=describe.Gradient("b"),
)
class LayerNorm(Model):
    name = "layernorm"

    def __init__(self, child=None, **kwargs):
        self.child = child
        if child is not None:
            self._layers = [child]
        else:
            self._layers = []
        Model.__init__(self, **kwargs)
        if "nO" in kwargs:
            self.nO = kwargs["nO"]
        elif getattr(child, "nO", None):
            self.nO = child.nO
        self.nr_upd = 0
        self.on_data_hooks.append(util.run_child_hooks)

    def predict(self, X):
        if X.size == 0:
            return X
        if self.child is not None:
            X = self.child.predict(X)
        N, mu, var = _get_moments(self.ops, X)
        Xh = _forward(self.ops, X, mu, var)
        y = Xh * self.G + self.b
        return y

    def begin_update(self, X):
        if self.child is not None:
            X, backprop_child = self.child.begin_update(X)
        else:
            backprop_child = None
        N, mu, var = _get_moments(self.ops, X)

        Xhat = _forward(self.ops, X, mu, var)

        y, backprop_rescale = self._begin_update_scale_shift(Xhat)

        def finish_update(dy):
            dy = backprop_rescale(dy)
            dist, sum_dy, sum_dy_dist = _get_d_moments(self.ops, dy, X, mu)
            d_xhat = N * dy - sum_dy - dist * var ** (-1.0) * sum_dy_dist
            d_xhat *= var ** (-1.0 / 2)
            d_xhat /= N
            if backprop_child is not None:
                return backprop_child(d_xhat)
            else:
                return d_xhat

        assert y.dtype == "float32"

        return y, finish_update

    def _begin_update_scale_shift(self, input__BI):
        def finish_update(gradient__BI):
            self.d_b += gradient__BI.sum(axis=0)
            d_G = self.d_G
            d_G += (gradient__BI * input__BI).sum(axis=0)
            return gradient__BI * self.G

        return input__BI * self.G + self.b, finish_update


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


def _forward(ops, X, mu, var):
    return (X - mu) * var ** (-1.0 / 2.0)
