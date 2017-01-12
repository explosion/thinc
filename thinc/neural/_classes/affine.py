from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases
from ..exceptions import ShapeError
from ..mem import Memory


def _set_dimensions_if_given(model, *args, **kwargs):
    if len(args) >= 1:
        model.nO = args[0]
    elif not hasattr(model, 'nO'):
        model.nO = None
    elif len(args) >= 2:
        model.nI = args[1]
    elif not hasattr(model, 'nI'):
        model.nI = None


def _alloc_mem_if_able(model, *args, **kwargs):
    weights = [desc for (name, desc) in model.descriptions.items()
               if isinstance(desc, describe.Weights)]
    shapes = {}
    for param in weights:
        shape = tuple([getattr(model, dim, None) for dim in param.shape])
        if any(dim is None for dim in shape):
            return None
        shapes[param.name] = shape
    for name, shape in shapes.items():
        model.mem.add(name, shape)


def _init_weights_if_able(model, *args, **kwargs):
    if model.mem:
        for name, shape, init in model.description.weights:
            weights = model.weights[name]
            init(weights)


def _set_dimensions_and_weights_if_needed(model, X, y=None):
    if model.dims.nI is None:
        model.dims.nI = X.shape[0]
    if model.dims.nO is None and y is not None:
        model.dims.nO = y.max()
    if model.mem is None:
        _alloc_mem_if_able(model)
        _init_weights_if_able(model)


@describe.input(("nB", "nI"))
@describe.output(("nB", "nO"))
@describe.on_data(_set_dimensions_and_weights_if_needed)
@describe.on_init(
    _set_dimensions_if_given,
    _alloc_mem_if_able,
    _init_weights_if_able)
@describe.attributes(
    nB=Dimension("Batch size"),
    nI=Dimension("Input size"),
    nO=Dimension("Output size"),
    W=Synapses("Weights matrix", ("nO", "nI"), lambda W, ops: ops.xavier_init(W)),
    b=Biases("Bias vector", ("nO",))
)
class Affine(Model):
    '''Computes the linear transform Y = (W @ X) + b.'''
    name = 'affine'

    def predict(self, input__BI):
        return self.ops.affine(self.W, self.b, input__BI)

    def begin_update(self, input_BI):
        output_BO = self.predict(input_BI)
        def finish_update(grad__BO):
            self.grads.W += self.ops.batch_outer(grad__BO, input__BI)
            self.grads.b += grad__BO.sum(axis=0)
            return self.ops.batch_dot(grad__BO, self.W.T)
        return output__BO, finish_update
