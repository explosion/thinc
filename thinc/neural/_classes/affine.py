from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases, Gradient
from ..mem import Memory
from ... import check
from ...check import has_shape


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
        lambda W, ops: ops.xavier_uniform_init(W)),
    b=Biases("Bias vector",
        lambda obj: (obj.nO,)),
    d_W=Gradient("W"),
    d_b=Gradient("b")
)
class Affine(Model):
    '''Computes the linear transform Y = (W @ X) + b.'''
    name = 'affine'

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

    @check.arg(1, has_shape(('nB', 'nI')))
    def predict(self, input__BI):
        return self.ops.affine(self.W, self.b, input__BI)

    @check.arg(1, has_shape(('nB', 'nI')))
    def begin_update(self, input__BI, drop=0.):
        output__BO = self.predict(input__BI)
        def finish_update(grad__BO, sgd=None):
            self.d_W += self.ops.batch_outer(grad__BO, input__BI)
            self.d_b += grad__BO.sum(axis=0)
            grad__BI = self.ops.dot(grad__BO, self.W)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient,
                    key=self.id)
            return grad__BI
        drop *= self.drop_factor
        output__BO, bp_dropout = self.ops.dropout(output__BO, drop, inplace=True)
        return output__BO, bp_dropout(finish_update)
