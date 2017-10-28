import numpy
from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases, Gradient
from ..util import get_array_module
from ..util import copy_array


def _set_dimensions_if_needed(model, X, y=None):
    if model.nI is None:
        model.nI = X.shape[1]
    if model.nO is None and y is not None: # pragma: no cover
        model.nO = int(y.max()) + 1


def xavier_uniform_init(W, ops):
    if (W**2).sum() != 0:
        return
    xp = get_array_module(W)
    scale = xp.sqrt(6. / (W.shape[0] + W.shape[2]))
    shape = (W.shape[0], W.shape[2])
    for i in range(W.shape[1]):
        xp.copyto(W[:,i], xp.random.uniform(-scale, scale, shape))


def normal_init(W, ops):
    if (W**2).sum() != 0:
        return
    xp = get_array_module(W)
    scale = xp.sqrt(1. / W.shape[-1])
    shape = (W.shape[0], W.shape[-1])
    size = xp.prod(shape)
    for i in range(W.shape[1]):
        xp.copyto(W[:,i], xp.random.normal(loc=0, scale=scale, size=size).reshape(shape))

@describe.on_data(_set_dimensions_if_needed)
@describe.output(("nO",))
@describe.input(("nI",))
@describe.attributes(
    nI=Dimension("Size of input"),
    nP=Dimension("Number of pieces"),
    nO=Dimension("Size of output"),
    W=Synapses("The weights matrix", lambda obj: (obj.nO, obj.nP, obj.nI),
        xavier_uniform_init),
    b=Biases("Bias parameter", lambda obj: (obj.nO, obj.nP)),
    d_W=Gradient("W"),
    d_b=Gradient("b")
)
class Maxout(Model):
    name = 'maxout'
    def __init__(self, nO=None, nI=None, pieces=2, **kwargs):
        Model.__init__(self, **kwargs)
        self.nO = nO
        self.nI = nI
        self.nP = pieces
        self.drop_factor = kwargs.get('drop_factor', 1.0)

    def predict(self, X__BI):
        W = self.W.reshape((self.nO * self.nP, self.nI))
        X__BOP = self.ops.batch_dot(X__BI, W)
        X__BOP += self.b.reshape((self.nO*self.nP,))
        X__BOP = X__BOP.reshape((X__BOP.shape[0], self.nO, self.nP))
        best__BO, _ = self.ops.maxout(X__BOP)
        return best__BO

    def begin_update(self, X__bi, drop=0.):
        W = self.W.reshape((self.nO * self.nP, self.nI))
        drop *= self.drop_factor
        output__boc = self.ops.batch_dot(X__bi, W)
        output__boc += self.b.reshape((self.nO*self.nP,))
        output__boc = output__boc.reshape((output__boc.shape[0], self.nO, self.nP))
        best__bo, which__bo = self.ops.maxout(output__boc)
        best__bo, bp_dropout = self.ops.dropout(best__bo, drop)

        def finish_update(dX__bo, sgd=None):
            dX__bop = self.ops.backprop_maxout(dX__bo, which__bo, self.nP)
            self.d_b += dX__bop.sum(axis=0)
            dX__bop = dX__bop.reshape((dX__bop.shape[0], self.nO*self.nP))
            d_W = self.ops.batch_outer(dX__bop, X__bi)
            self.d_W += d_W.reshape((self.nO, self.nP, self.nI))
            # Bop,opi->Bi
            dX__bi = self.ops.dot(dX__bop, self.W.reshape((self.nO*self.nP, self.nI)))
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return dX__bi
        return best__bo, bp_dropout(finish_update)
