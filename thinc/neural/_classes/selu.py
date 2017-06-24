from .affine import Affine
from ... import describe
from ...describe import Dimension, Synapses, Biases
from ... import check
from ...describe import Dimension, Synapses, Biases
from ...check import has_shape

from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases, Gradient
from ..mem import Memory
from ... import check
from ...check import has_shape
from .._lsuv import LSUVinit
from ..util import copy_array


def _set_dimensions_if_needed(model, X, y=None):
    if model.nI is None:
        model.nI = X.shape[1]
    if model.nO is None and y is not None:
        if len(y.shape) == 2:
            model.nO = y.shape[1]
        else:
            model.nO = int(y.max()) + 1


@describe.on_data(_set_dimensions_if_needed)
@describe.attributes(
    nB=Dimension("Batch size"),
    nI=Dimension("Input size"),
    nO=Dimension("Output size"),
    W=Synapses("Weights matrix",
        lambda obj: (obj.nO, obj.nI),
        lambda W, ops: ops.normal_init(W, W.shape[-1])),
    b=Biases("Bias vector",
        lambda obj: (obj.nO,)),
    d_W=Gradient("W"),
    d_b=Gradient("b")
)
class SELU(Model):
    name = 'selu'

    @property
    def input_shape(self):
        return (self.nB, self.nI)

    @property
    def output_shape(self):
        return (self.nB, self.nO)

    def __init__(self, nO=None, nI=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.nO = nO
        self.nI = nI
        self.drop_factor = kwargs.get('drop_factor', 1.0)

    def predict(self, input__bi):
        output__bo = self.ops.affine(self.W, self.b, input__bi)
        self.ops.selu(output__bo, inplace=True)
        return output__bo

    def begin_update(self, input__bi, drop=0.):
        output__bo = self.predict(input__bi)
        output_copy = self.ops.xp.ascontiguousarray(output__bo, dtype='f')
        self.ops.selu(output_copy, inplace=True)
        def finish_update(grad__bo, sgd=None):
            grad__bo = self.ops.xp.ascontiguousarray(grad__bo, dtype='f')
            self.ops.backprop_selu(grad__bo, output_copy, inplace=True)
            self.d_W += self.ops.batch_outer(grad__bo, input__bi)
            self.d_b += grad__bo.sum(axis=0)
            grad__BI = self.ops.batch_dot(grad__bo, self.W.T)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return grad__BI
        drop *= self.drop_factor
        return self.dropout(output__bo, finish_update, drop)

    def dropout(self, X, finish_update, drop):
        if drop <= 0:
            return X, finish_update
        alpha = -1.75809934
        q = 1. - drop
        a = (q * (1. + alpha * alpha * (1.-q))) ** -0.5
        b = -a * (1.-q) * alpha
        mask = self.ops.xp.random.uniform(0., 1., X.shape)
        def backprop_selu_dropout(d_dropped, sgd=None):
            return finish_update(a * d_dropped * mask, sgd=sgd)
        dropped = self.ops.xp.where(mask >= drop, X, alpha)
        return a * dropped + b, backprop_selu_dropout
