from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases, Gradient
from ..exceptions import ShapeError
from ..mem import Memory


def _set_dimensions_if_given(model, *args, **kwargs):
    if len(args) >= 1:
        model.nO = args[0]
    elif not hasattr(model, 'nO'):
        model.nO = None
    if len(args) >= 2:
        model.nI = args[1]
    elif not hasattr(model, 'nI'):
        model.nI = None


def _set_dimensions_if_needed(model, X, y=None):
    if model.nI is None:
        model.nI = X.shape[0]
    if model.nO is None and y is not None:
        model.nO = y.max()


@describe.input(lambda obj, **_: (obj.nB, obj.nI))
@describe.output(lambda obj, **_: (obj.nB, obj.nO))
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
@describe.on_init(_set_dimensions_if_given)
@describe.on_data(_set_dimensions_if_needed)
class Affine(Model):
    '''Computes the linear transform Y = (W @ X) + b.'''
    name = 'affine'

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
