from .model import Model
from ... import describe


def _init_to_one(W, ops):
    W[:] = 1.

def _run_child_hooks(model, X, y=None):
    for hook in model.child.on_data_hooks:
        hook(model.child, X, y)


@describe.on_data(_run_child_hooks)
@describe.attributes(
    G=describe.Weights("Scaling vector",
        lambda obj: (obj.child.nO,), _init_to_one),
    b=describe.Biases("Bias vector",
        lambda obj: (obj.child.nO,)),
    d_G=describe.Gradient("G"),
    d_b=describe.Gradient("b")
)
class BatchNorm(Model):
    name = 'batchnorm'

    @property
    def input_shape(self):
        return self.child.input_shape

    @property
    def output_shape(self):
        return self.child.output_shape

    def __init__(self, child, **kwargs):
        self.child = child
        self._layers = [child]
        Model.__init__(self, **kwargs)

    def predict(self, X):
        X = self.child.predict(X)
        N, mu, var = _get_moments(self.ops, X)
        Xh = _forward(self.ops, X, mu, var)
        y = Xh * self.G + self.b
        return y
 
    def begin_update(self, X, drop=0.):
        X, backprop_child = self.child.begin_update(X, drop=0.) # Steal dropout
        N, mu, var = _get_moments(self.ops, X)
        Xhat = _forward(self.ops, X, mu, var)
        y, backprop_rescale = self._begin_update_scale_shift(Xhat)

        def finish_update(dy, sgd=None):
            dy = backprop_rescale(dy, sgd)
            dist, sum_dy, sum_dy_dist = _get_d_moments(self.ops, dy, X, mu)
            d_xhat = N * dy - sum_dy - dist * var**(-1.) * sum_dy_dist
            d_xhat *= var ** (-1. / 2)
            d_xhat /= N
            return backprop_child(d_xhat, sgd)
        y, bp_dropout = self.ops.dropout(y, drop, inplace=True)
        return y, bp_dropout(finish_update)

    def _begin_update_scale_shift(self, input__BI):
        def finish_update(gradient__BI, sgd=None):
            self.d_b += gradient__BI.sum(axis=0)
            for i in range(gradient__BI.shape[0]):
                self.d_G += gradient__BI[i] * input__BI[i]
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=id(self._mem))
            return gradient__BI * self.G
        return input__BI * self.G + self.b, finish_update


def _get_moments(ops, X):
    if hasattr(X, 'shape') and len(X.shape) == 2:
        mu = X.mean(axis=0)
        var = X.var(axis=0) + 1e-8
        return X.shape[0], mu, var
    else:
        stacked = numpy.vstack(X)
        return stacked.shape[0], stacked.mean(axis=0), stacked.var(axis=0)


def _get_d_moments(ops, dy, X, mu):
    if hasattr(dy, 'shape'):
        dist = X-mu
        return dist, ops.xp.sum(dy, axis=0), ops.xp.sum(dy * dist, axis=0)
    else:
        sum_dy = [ops.xp.sum(seq, axis=0) for seq in dy]
        dist = [x-mu for x in X]
        sum_dy_dot_dist = [ops.xp.sum(seq * d, axis=0) for seq, d in zip(dy, dist)]
        return dist, sum_dy, sum_dy_dot_dist


def _forward(ops, X, mu, var):
    if hasattr(X, 'shape'):
        return (X-mu) * var ** (-1./2.)
    else:
        return [_forward(x, mu, var) for x in X]
