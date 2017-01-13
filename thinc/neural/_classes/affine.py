from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases, Gradient
from ..exceptions import ShapeError
from ..mem import Memory


def _set_dimensions_if_needed(model, X, y=None):
    if model.nI is None:
        model.nI = X.shape[0]
    if model.nO is None and y is not None:
        model.nO = y.max()


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
@describe.on_data(_set_dimensions_if_needed)
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

    def predict(self, input__BI):
        return self.ops.affine(self.W, self.b, input__BI)

    def begin_update(self, input__BI):
        output__BO = self.predict(input__BI)
        def finish_update(grad__BO):
            self.d_W += self.ops.batch_outer(grad__BO, input__BI)
            self.d_b += grad__BO.sum(axis=0)
            return self.ops.batch_dot(grad__BO, self.W.T)
        return output__BO, finish_update

    def apply_updates(self, optimizer):
        optimizer(self.W, self.d_W, key=(self.id, 'W'))
        optimizer(self.b, self.d_b, key=(self.id, 'b'))
