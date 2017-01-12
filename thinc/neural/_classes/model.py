from numpy import prod
import contextlib

from .. import util
from ..train import Trainer
from ..exceptions import ShapeError
from ..ops import Ops
from ..params import Params


def allocate_params(model):
    '''Allocate all trainable parameters of a model, including for its sublayers,
    so that parameters can be contiguous in memory.'''
    # Get total size
    size = sum(prod(shape) for shape in model.describe_all_params)
    params = Params(total=size)
    for i, layer in enumerate(all_layers(model)):
        layer.name = '%s-%d' % (layer.name, i)
        for name, shape, init in describe_params(model):
            params.add(name, shape)
            if init is not None:
                init(params.get(name), inplace=True)
    return params


def set_dimensions_given_data(model, X, y):
    pass


def initialize_weights_given_data(model, X, y):
    pass


def all_layers(model):
    '''Iterate over all layers in the model.'''
    queue = [model]
    seen = set()
    for layer in queue:
        yield layer
        queue.extend(layer.layers)
        assert id(layer) not in seen # Catch loops!
        seen.add(id(layer))


def describe_all_params(model):
    for layer in model.all_layers:
        for name, shape, init in layer.describe_params:
            yield name, shape, init


class Model(object):
    '''Model base class.'''
    dims = None
    weights = None
    layers = None
    mem = None
    Trainer = Trainer

    @classmethod
    @contextlib.contextmanager
    def bind_operators(cls, ops):
        old_ops = dict(cls._operators)
        for op, func in ops.items():
            cls._operators[op] = func
        yield
        cls._operators = old_ops

    def __init__(self, *args, **kwargs):
        self.args = {i: arg for i, arg in enumerate(args)}
        self.args.update(kwargs)
        self.ops = kwargs.get('ops', NumpyOps())
        self._layers = []
        self._operators = {}
        for name in self.desc.dimensions:
            if not hasattr(self, name):
                setattr(self, name, None)
        self.desc.dimensions.set_from_init(self.args)

    def begin_training(self, train_X, train_Y):
        set_dimensions_given_data(self, train_X, train_Y)
        self.params = allocate_params(self)
        initialize_weights_given_data(self, train_X, train_Y)
        return self.Trainer(self, train_X, train_Y)
 
    def predict(self, X):
        y, _ = self.begin_update(X)
        return y
  
    def begin_update(self, X, **kwargs):
        self.check_input(X, expect_batch=True)
        callbacks = []
        for layer in self.layers:
            X = self.ops.xp.ascontiguousarray(X, dtype='f')
            X, inc_layer_grad = layer.begin_update(X)
            callbacks.append(inc_layer_grad)
        def continue_update(gradient):
            for callback in reversed(callbacks):
                gradient = callback(gradient)
            return gradient
        return X, continue_update

    def __add__(self, other):
        '''Apply the function bound to the '+' operator.'''
        return self._operators['+'](self, other)

    def __sub__(self, other):
        '''Apply the function bound to the '-' operator.'''
        return self._operators['-'](self, other)

    def __mul__(self, other):
        '''Apply the function bound to the '*' operator.'''
        return self._operators['*'](self, other)

    def __matmul__(self, other):
        '''Apply the function bound to the '@' operator.'''
        return self._operators['@'](self, other)

    def __truediv__(self, other):
        '''Apply the function bound to the '/' operator.'''
        return self._operators['/'](self, other)

    def __floordiv__(self, other):
        '''Apply the function bound to the '//' operator.'''
        return self._operators['//'](self, other)

    def __mod__(self, other):
        '''Apply the function bound to the '%' operator.'''
        return self._operators['%'](self, other)

    def __pow__(self, other, modulo=None):
        '''Apply the function bound to the '**' operator.'''
        return self._operators['**'](self, other)

    def __lshift__(self, other):
        '''Apply the function bound to the '<<' operator.'''
        return self._operators['<<'](self, other)

    def _rshift__(self, other):
        '''Apply the function bound to the '>>' operator.'''
        return self._operators['>>'](self, other)

    def __and__(self, other):
        '''Apply the function bound to the '&' operator.'''
        return self._operators['&'](self, other)

    def __xor__(self, other):
        '''Apply the function bound to the '^' operator.'''
        return self._operators['^'](self, other)

    def __or__(self, other):
        '''Apply the function bound to the '|' operator.'''
        return self._operators['|'](self, other)

#
#    def predict_one(self, x):
#        X = self.ops.expand_dims(x, axis=0)
#        return self.predict_batch(X)[0]
#
#
##    
#    def pipe(self, stream, batch_size=1000):
#        for batch in util.minibatch(stream, batch_size):
#            ys = self.predict_batch(batch)
#            for y in ys:
#                yield y
#
#    def update(self, stream, batch_size=1000):
#        for X, y in util.minibatch(stream, batch_size=batch_size):
#            output, finish_update = self.begin_update(X)
#            gradient = finish_update(y)
#            yield gradient
# 
#def list_gradients(self):
#    pass
#
#@contextlib.contextmanager
#def use_params(self, params):
#    if ('', self.name) in params:
#        current = self.params.weights.copy()
#        current[:] = self.params.weights
#        self.params.weights[:] = params[('', self.name)]
#        yield
#        self.params.weights[:] = current
#

#def _update_defaults(self, *args, **kwargs):
#    new_kwargs = {}
#    for key, value in kwargs.items():
#        if hasattr(self, key):
#            setattr(self, key, value)
#        else:
#            new_kwargs[key] = value
#    return new_kwargs
#
#def check_input(self, x, expect_batch=False):
#    if self.layers:
#        return self.layers[0].check_input(x, expect_batch=expect_batch)
#    if is_batch(x):
#        shape = x.shape[1:]
#    else:
#        raise ShapeError.expected_batch(locals(), globals())
#    if shape != self.input_shape:
#        raise ShapeError.dim_mismatch(self.input_shape, shape)
#    else:
#        return True
#
#def __call__(self, x):
#    '''Predict a single x.'''
#    self.check_input(x)
#    if is_batch:
#        return self.predict_batch(x)
#    else:
#        return self.predict_one(x)
#
#
