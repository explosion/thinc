from numpy import prod
import contextlib

from .. import util
from ..train import Trainer
from ..exceptions import ShapeError
from ..ops import Ops
from ..params import Params


def is_batch(x):
    return True
    #return isinstance(x, list) or len(x.shape) >= 2


class Model(object):
    '''Model base class.'''
    name = 'model'
    device = 'cpu'
    Trainer = Trainer
    ops = None
    layers = []
    Params = Params

    @property
    def size(self):
        if self.shape is None:
            return None
        else:
            return prod(self.shape)

    @property
    def describe_params(self):
        for desc in []: # Need to be empty generator
            yield desc
    
    @property
    def input_shape(self):
        return self.layers[0].input_shape if self.layers else None
 
    @property
    def output_shape(self):
        return self.layers[0].input_shape if self.layers else None

    @property
    def shape(self):
        return self.output_shape + self.input_shape if self.layers else None

    def __init__(self, *layers, **kwargs):
        self.layers = []
        kwargs = self._update_defaults(**kwargs)
        if self.ops is None:
            self.ops = util.get_ops(self.device)
        self._root = kwargs.get('root', self)
        self._parent = kwargs.get('parent', None)
        if self._parent is not None:
            self.name = self._parent.name + '-' + self.name
        self.params = self.Params(self.ops)

        kwargs.setdefault('root', self)
        kwargs['parent'] = self
        for layer in layers:
            if isinstance(layer, Model):
                self.layers.append(layer)
            else:
                self.layers.append(layer(**kwargs))

    def _update_defaults(self, *args, **kwargs):
        new_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                new_kwargs[key] = value
        return new_kwargs
    
    def initialize_params(self, train_data=None, add_gradient=True):
        for name, shape, init in self.describe_params:
            if name not in self.params:
                self.params.add(name, shape)
                if init is not None:
                    init(self.params.get(name), inplace=True)
        for layer in self.layers:
            layer.params = self.params
            layer.initialize_params(train_data, add_gradient=add_gradient)

    def check_input(self, x, expect_batch=False):
        if self.layers:
            return self.layers[0].check_input(x, expect_batch=expect_batch)
        if is_batch(x):
            shape = x.shape[1:]
        else:
            raise ShapeError.expected_batch(locals(), globals())
        if shape != self.input_shape:
            raise ShapeError.dim_mismatch(self.input_shape, shape)
        else:
            return True

    def __call__(self, x):
        '''Predict a single x.'''
        self.check_input(x)
        if is_batch:
            return self.predict_batch(x)
        else:
            return self.predict_one(x)

    def predict_one(self, x):
        X = self.ops.expand_dims(x, axis=0)
        return self.predict_batch(X)[0]


    def predict_batch(self, X):
        y, _ = self.begin_update(X)
        return y

    def begin_training(self, train_data):
        self.initialize_params(train_data, add_gradient=True)
        return self.Trainer(self, train_data)
    
    def pipe(self, stream, batch_size=1000):
        for batch in util.minibatch(stream, batch_size):
            ys = self.predict_batch(batch)
            for y in ys:
                yield y

    def update(self, stream, batch_size=1000):
        for X, y in util.minibatch(stream, batch_size=batch_size):
            output, finish_update = self.begin_update(X)
            gradient = finish_update(y)
            yield gradient
    
    def begin_update(self, X, **kwargs):
        self.check_input(X, expect_batch=True)
        callbacks = []
        for layer in self.layers:
            X = self.ops.xp.ascontiguousarray(X, dtype='f')
            X, finish_update = layer.begin_update(X, **kwargs)
            callbacks.append(finish_update)
        return X, self._get_finish_update(callbacks)
    
    def _get_finish_update(self, callbacks):
        def finish_update(gradient, optimizer=None, **kwargs):
            for callback in reversed(callbacks):
                gradient = self.ops.xp.ascontiguousarray(gradient, dtype='f')
                gradient = callback(gradient, optimizer=optimizer,
                                is_child=True)
            if optimizer is not None and self.params is not None \
            and not kwargs.get('is_child'):
                optimizer(self.params.weights, self.params.gradient,
                    key=('', self.name), **kwargs)
            return gradient
        return finish_update

    @contextlib.contextmanager
    def use_params(self, params):
        if ('', self.name) in params:
            current = self.params.weights.copy()
            current[:] = self.params.weights
            self.params.weights[:] = params[('', self.name)]
            yield
            self.params.weights[:] = current
