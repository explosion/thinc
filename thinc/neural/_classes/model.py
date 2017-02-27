from __future__ import division
from numpy import prod
import contextlib

from .. import util
from ..train import Trainer
from ..ops import NumpyOps
from ..mem import Memory
from ..util import get_ops
from ... import check
from ... import describe
from ...check import equal_length, has_shape, is_sequence, is_float, is_array


class Model(object):
    '''Model base class.'''
    name = 'model'
    id = 0
    ops = NumpyOps()
    Trainer = Trainer
    descriptions = []
    on_data_hooks = []
    on_init_hooks = [] # Use this to add layers
    _operators = {}

    @classmethod
    @contextlib.contextmanager
    def define_operators(cls, operators):
        '''Bind operators to specified functions for the scope of the context:

        Example
        -------

            model = Model()
            other = Model()
            with Model.define_operators({"+": lambda self, other: "plus"}):
                print(model + other)
                # "plus"
            print(model + other)
            # Raises TypeError --- binding limited to scope of with block.
        '''
        old_ops = dict(cls._operators)
        for op, func in operators.items():
            cls._operators[op] = func
        yield
        cls._operators = old_ops

    @classmethod
    @contextlib.contextmanager
    def use_device(cls, device):
        '''Change the device to execute on for the scope of the block.'''
        if device == cls.ops.device:
            yield
        else:
            curr_ops = cls.ops
            cls.ops = get_ops(device)
            yield
            cls.ops = curr_ops

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        self.name = self.__class__.name
        kwargs = self._update_defaults(args, kwargs)
        self._mem = Memory(self.ops)
        self._dims = {}
        if not hasattr(self, '_layers'):
            self._layers = []
        self.descriptions = dict(self.descriptions)
        self.on_init_hooks = list(self.on_init_hooks)
        self.on_data_hooks = list(self.on_data_hooks)

        for attr, install in self.descriptions.items():
            install(attr, self)
        for hook in self.on_init_hooks:
            hook(self, *args, **kwargs)
        self.set_id()

    def _update_defaults(self, args, kwargs):
        new_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                new_kwargs[key] = value
        return new_kwargs

    def set_id(self):
        Model.id += 1
        self.id = Model.id
        for layer in self._layers:
            layer.set_id()

    #@check.args(equal_length)
    @check.arg(1, is_sequence)
    def begin_training(self, train_X, train_y=None, **trainer_cfg):
        for hook in self.on_data_hooks:
            hook(self, train_X, train_y)
        return self.Trainer(self, **trainer_cfg)

    @check.arg(2, is_float)
    @check.arg(1, has_shape(('nB', 'nI')))
    def begin_update(self, X, drop=0.0):
        raise NotImplementedError

    def predict(self, X):
        y, _ = self.begin_update(X)
        return y

    def predict_one(self, x):
        X = self.ops.expand_dims(x, axis=0)
        return self.predict(X)[0]

    @contextlib.contextmanager
    def use_params(self, params): # pragma: no cover
        backup = None
        weights = self._mem.weights
        if self.id in params:
            param = params[self.id]
            backup = weights.copy()
            weights[:] = param
        if hasattr(self, '_layers'):
            contexts = [layer.use_params(params) for layer in self._layers]
            for context in contexts:
                next(context.gen)
        yield
        if backup is not None:
            weights[:] = backup
        for context in contexts:
            next(context.gen)

    def __call__(self, x):
        '''
        x
            Must match expected type
            Must match expected shape
        '''
        return self.predict(x)

    def pipe(self, stream, batch_size=1000):
        for batch in util.minibatch(stream, batch_size):
            ys = self.predict(batch)
            for y in ys:
                yield y

    def update(self, stream, batch_size=1000):
        for X, y in util.minibatch(stream, batch_size=batch_size):
            output, finish_update = self.begin_update(X)
            gradient = finish_update(y)
            yield gradient

    def evaluate(self, X, y):
        '''
        x
            Must match expected type
            Must match expected shape
        y
            Must match expected type
        '''
        scores = self.ops.xp.vstack(self.pipe(X))
        scores = scores.reshape(y.shape)
        if len(scores.shape) == 1:
            correct = ((scores >= 0.5) == (y >= 0.5)).sum()
        else:
            correct = (scores.argmax(axis=1) == y.argmax(axis=1)).sum()
        return correct / y.shape[0]

    @check.operator_is_defined('+')
    def __add__(self, other):
        '''Apply the function bound to the '+' operator.'''
        return self._operators['+'](self, other)

    @check.operator_is_defined('-')
    def __sub__(self, other):
        '''Apply the function bound to the '-' operator.'''
        return self._operators['-'](self, other)

    @check.operator_is_defined('*')
    def __mul__(self, other):
        '''Apply the function bound to the '*' operator.'''
        return self._operators['*'](self, other)

    @check.operator_is_defined('@')
    def __matmul__(self, other):
        '''Apply the function bound to the '@' operator.'''
        return self._operators['@'](self, other)

    @check.operator_is_defined('/')
    def __div__(self, other):
        '''Apply the function bound to the '/' operator.'''
        return self._operators['/'](self, other)

    @check.operator_is_defined('/')
    def __truediv__(self, other): # pragma: no cover
        '''Apply the function bound to the '/' operator.'''
        return self._operators['/'](self, other)

    @check.operator_is_defined('//')
    def __floordiv__(self, other):
        '''Apply the function bound to the '//' operator.'''
        return self._operators['//'](self, other)

    @check.operator_is_defined('%')
    def __mod__(self, other):
        '''Apply the function bound to the '%' operator.'''
        return self._operators['%'](self, other)

    @check.operator_is_defined('**')
    def __pow__(self, other, modulo=None):
        '''Apply the function bound to the '**' operator.'''
        return self._operators['**'](self, other)

    @check.operator_is_defined('<<')
    def __lshift__(self, other):
        '''Apply the function bound to the '<<' operator.'''
        return self._operators['<<'](self, other)

    @check.operator_is_defined('>>')
    def __rshift__(self, other):
        '''Apply the function bound to the '>>' operator.'''
        return self._operators['>>'](self, other)

    @check.operator_is_defined('&')
    def __and__(self, other):
        '''Apply the function bound to the '&' operator.'''
        return self._operators['&'](self, other)

    @check.operator_is_defined('^')
    def __xor__(self, other):
        '''Apply the function bound to the '^' operator.'''
        return self._operators['^'](self, other)

    @check.operator_is_defined('|')
    def __or__(self, other):
        '''Apply the function bound to the '|' operator.'''
        return self._operators['|'](self, other)

#def list_gradients(self):
#    pass
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
#def allocate_params(model):
#    '''Allocate all trainable parameters of a model, including for its sublayers,
#    so that parameters can be contiguous in memory.'''
#    # Get total size
#    size = sum(prod(shape) for shape in model.describe_all_params)
#    params = Params(total=size)
#    for i, layer in enumerate(all_layers(model)):
#        layer.name = '%s-%d' % (layer.name, i)
#        for name, shape, init in describe_params(model):
#            params.add(name, shape)
#            if init is not None:
#                init(params.get(name), inplace=True)
#    return params
#
#
#def set_dimensions_given_data(model, X, y):
#    pass
#
#
#def initialize_weights_given_data(model, X, y):
#    for name, weights in model.weights.items():
#        init = model.get_initialzer(name)
#
#    for key, (name, shape, init_func) in model.description.weights:
#        if init_func is not None:
#            weights = model.w.get(key)
#            init_func(weights, X, y)
#
#
#
#def all_layers(model):
#    '''Iterate over all layers in the model.'''
#    queue = [model]
#    seen = set()
#    for layer in queue:
#        yield layer
#        queue.extend(layer.layers)
#        assert id(layer) not in seen # Catch loops!
#        seen.add(id(layer))
#
#
#def describe_all_params(model):
#    for layer in model.all_layers:
#        for name, shape, init in layer.describe_params:
#            yield name, shape, init
#
#
