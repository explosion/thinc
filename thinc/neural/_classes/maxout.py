import numpy
from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases, Gradient
from .._lsuv import LSUVinit


def _set_dimensions_if_needed(model, X, y=None):
    if model.nI is None:
        model.nI = X.shape[1]
    if model.nO is None and y is not None: # pragma: no cover
        model.nO = int(y.max()) + 1


def xavier_uniform_init(W, ops):
    scale = ops.xp.sqrt(6. / (W.shape[0] + W.shape[2]))
    shape = (W.shape[0], W.shape[2])
    for i in range(W.shape[1]):
        ops.xp.copyto(W[:,i], ops.xp.random.uniform(-scale, scale, shape))


@describe.on_data(_set_dimensions_if_needed, LSUVinit)
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

    def predict(self, X__BI):
        X__BOP = self.ops.xp.tensordot(X__BI, self.W, axes=[[1], [-1]])
        X__BOP += self.b
        best__BO, _ = self.ops.maxout(X__BOP)
        return best__BO

    def begin_update(self, X__bi, drop=0.):
        output__boc = self.ops.xp.tensordot(X__bi, self.W, axes=[[1], [-1]])
        output__boc += self.b
        best__bo, which__bo = self.ops.maxout(output__boc)
        best__bo, bp_dropout = self.ops.dropout(best__bo, drop, inplace=True)
 
        def finish_update(dX__bo, sgd=None):
            dX__bop = self.ops.backprop_maxout(dX__bo, which__bo, self.nP)
            self.d_b += dX__bop.sum(axis=0)
            self.d_W += self.ops.xp.tensordot(dX__bop, X__bi, axes=[[0], [0]])
            # Bop,opi->Bi
            dX__bi = self.ops.xp.tensordot(dX__bop, self.W, axes=[[1,2], [0, 1]])
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return dX__bi
        return best__bo, bp_dropout(finish_update)
