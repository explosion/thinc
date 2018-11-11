from ... import describe
from .model import Model

REPRODUCE_BUG = False

def set_compat_six_eight(flag_value):
    '''Allow backwards compatibility with calculations bug from Thinc 6.8'''
    global REPRODUCE_BUG
    REPRODUCE_BUG = flag_value


def _init_to_one(W, ops):
    W.fill(1.)

def _run_child_hooks(model, X, y=None):
    for hook in model.child.on_data_hooks:
        hook(model.child, X, y)
    #model.nO = model.child.nO


@describe.on_data(_run_child_hooks)
@describe.attributes(
    G=describe.Weights("Scaling vector",
        lambda obj: (obj.nO,), _init_to_one),
    b=describe.Biases("Bias vector",
        lambda obj: (obj.nO,)),
    d_G=describe.Gradient("G"),
    d_b=describe.Gradient("b")
)
class LayerNorm(Model):
    name = 'layernorm'

    def __init__(self, child, **kwargs):
        self.child = child
        self._layers = [child]
        Model.__init__(self, **kwargs)
        if 'nO' in kwargs:
            self.nO = kwargs['nO']
        elif getattr(child, 'nO', None):
            self.nO = child.nO
        self.nr_upd = 0

    def predict(self, X):
        X = self.child.predict(X)
        out = self.ops.layer_norm(X)
        out *= self.G
        out += self.b
        return out

    def begin_update(self, X, drop=0.):
        X, backprop_child = self.child.begin_update(X, drop=0.)
        Xhat = self.ops.layer_norm(X)
        Y, backprop_rescale = self._begin_update_scale_shift(Xhat)

        def finish_update(dY, sgd=None):
            dY = backprop_rescale(dY, sgd)
            dXhat = self.ops.backprop_layer_norm(dY, X)
            return backprop_child(d_xhat, sgd)
        if drop is not None:
            drop *= getattr(self.child, 'drop_factor', self.ops.asarray([1.0], dtype='f'))
        Y, bp_dropout = self.ops.dropout(Y, drop)
        assert Y.dtype == 'float32'
        return Y, bp_dropout(finish_update)

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
    if REPRODUCE_BUG:
        return _get_moments_reproduce_bug(ops, X)
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True) + 1e-08
    return ops.asarray([X.shape[1]], dtype='f'), mu, var


def _get_moments_reproduce_bug(ops, X):
    '''Replicate bug from Thinc 6.8, for backwards compatibility.'''
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True) + 1e-08
    return ops.asarray([X.shape[0]], dtype='f'), mu, var



def _get_d_moments(ops, dy, X, mu):
    dist = X-mu
    return dist, ops.xp.sum(dy, axis=1, keepdims=True), ops.xp.sum(dy * dist, axis=1, keepdims=True)


def _forward(ops, X, mu, var):
    return (X-mu) * var ** (-1./2.)
