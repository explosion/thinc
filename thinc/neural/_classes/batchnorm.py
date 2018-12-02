# coding: utf8
from __future__ import unicode_literals

from .model import Model
from ... import describe


def _init_to_one(W, ops):
    W.fill(1.0)


def _run_child_hooks(model, X, y=None):
    for hook in model.child.on_data_hooks:
        hook(model.child, X, y)


@describe.on_data(_run_child_hooks)
@describe.attributes(
    G=describe.Weights("Scaling vector", lambda obj: (obj.nO,), _init_to_one),
    b=describe.Biases("Bias vector", lambda obj: (obj.nO,)),
    d_G=describe.Gradient("G"),
    d_b=describe.Gradient("b"),
    m=describe.Weights("Means", lambda obj: (obj.nO,)),
    v=describe.Weights("Variance", lambda obj: (obj.nO,), _init_to_one),
)
class BatchNorm(Model):
    name = "batchnorm"

    def __init__(self, child, **kwargs):
        self.child = child
        self._layers = [child]
        if "nO" in kwargs:
            self.nO = kwargs["nO"]
        elif getattr(child, "nO", None):
            self.nO = child.nO
        self.nr_upd = 0
        self.eps = kwargs.get("eps", 1e-5)
        self.alpha = self.ops.xp.asarray([0.1], dtype="float32")
        self.rmax = kwargs.get("rmax", 3.0)
        self.dmax = kwargs.get("dmax", 5.0)
        Model.__init__(self, **kwargs)

    def predict(self, X):
        X = self.child.predict(X)
        Xh = _forward(self.ops, X, self.m, self.v + self.eps)
        y = Xh * self.G + self.b
        return y

    def begin_update(self, X, drop=0.0):
        if drop is None:
            return self.predict(X), None
        assert X.dtype == "float32"
        X, backprop_child = self.child.begin_update(X, drop=0.0)
        N, mu, var = _get_moments(self.ops, X)
        var += self.eps
        r = self.ops.xp.clip(var / self.v, 1.0 / self.rmax, self.rmax)
        d = self.ops.xp.clip((mu - self.m) / self.v, -self.dmax, self.dmax)
        self.nr_upd += 1

        # I'm not sure this is the best thing to do --
        # Should we consider a sample be the instance, or the batch?
        # If we want the variance of the inputs it should be like:
        """
        diff = X - self.m
        incr = (1-alpha) * diff
        self.m += incr.mean(axis=0)
        self.v += (diff * incr).mean(axis=0)
        self.v *= alpha
        """
        self.m += self.alpha * (mu - self.m)
        self.v += self.alpha * (var - self.v)
        Xhat = _forward(self.ops, X, mu, var)
        Xhat *= r
        Xhat += d

        y, backprop_rescale = self._begin_update_scale_shift(Xhat)

        def finish_update(dy, sgd=None):
            dy = backprop_rescale(dy, sgd)
            dist, sum_dy, sum_dy_dist = _get_d_moments(self.ops, dy, X, mu)
            d_xhat = N * dy - sum_dy - dist * (1.0 / var) * sum_dy_dist
            d_xhat *= var ** (-1.0 / 2)
            d_xhat /= N
            return backprop_child(d_xhat, sgd)

        if drop is not None:
            drop *= getattr(self.child, "drop_factor", 1.0)
        y, bp_dropout = self.ops.dropout(y, drop)
        assert y.dtype == "float32"
        return y, bp_dropout(finish_update)

    def _begin_update_scale_shift(self, input__BI):
        def finish_update(gradient__BI, sgd=None):
            self.d_b += gradient__BI.sum(axis=0)
            d_G = self.d_G
            d_G += (gradient__BI * input__BI).sum(axis=0)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return gradient__BI * self.G

        return input__BI * self.G + self.b, finish_update


def _get_moments(ops, X):
    mu = X.mean(axis=0)
    var = X.var(axis=0) + 1e-08
    return ops.asarray([X.shape[0]], dtype="float32"), mu, var


def _get_d_moments(ops, dy, X, mu):
    dist = X - mu
    return dist, ops.xp.sum(dy, axis=0), ops.xp.sum(dy * dist, axis=0)


def _forward(ops, X, mu, var):
    return (X - mu) * var ** (-1.0 / 2.0)
