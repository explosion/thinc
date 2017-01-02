from numpy import prod

from . import util
from .util import Unassigned
from .train import Trainer
from .exceptions import ShapeError
from .ops import Ops
from .params import Params


class Model(object):
    '''Model base class.'''
    name = 'model'
    device = 'cpu'
    Trainer = Trainer
    ops = None
    output_shape = None
    input_shape = None
    layers = []
    params = None

    @property
    def size(self):
        if any(shape is None for name, shape, init in self.describe_params):
            return None
        return sum(prod(shape) for name, shape, init in self.describe_params)

    @property
    def describe_params(self):
        for desc in []: # Need to be empty generator
            yield desc

    @property
    def shape(self):
        if self.output_shape is not None and self.input_shape is not None:
            return self.output_shape + self.input_shape
        else:
            return None

    def __init__(self, *layers, **kwargs):
        self.layers = []
        kwargs = self._update_defaults(**kwargs)
        for layer in layers:
            if isinstance(layer, Model):
                self.layers.append(layer)
            else:
                self.layers.append(layer(**kwargs))
        if self.ops is None:
            self.ops = util.get_ops(self.device)

    def _update_defaults(self, *args, **kwargs):
        new_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                new_kwargs[key] = value
        return new_kwargs
    
    def initialize_params(self, train_data):
        shape = train_data[0].shape
        for i, dim in enumerate(shape):
            if self.shape[i] is None:
                self.shape[i] = dim
        for name, shape, init in self.describe_params:
            self.params.add(name, shape)
            init(self.params.get(name), train_data)
        for layer in self.layers:
            layer.initialize_params(train_data)
        self.params.merge_params(layer.params for layer in self.layers)

    def check_input(self, x):
        if is_batch(x):
            shape = x.shape[1:]
        if shape != self.input_shape:
            raise ShapeError.dim_mismatch(x, self)
        else:
            return True

    def __call__(self, x):
        '''Predict a single x.'''
        is_batch = self.is_batch(x)
        self.check_shape(x, is_batch)
        if is_batch:
            return self.predict_batch(x)
        else:
            return self.predict_one(x)

    def predict_one(self, x):
        X = self.ops.expand_dims(x, axis=0)
        return self.predict_batch(X)[0]

    def predict_batch(self, X):
        for layer in self.layers:
            X = layer.predict_batch(X)
        return X

    def begin_training(self, train_data):
        self.initialize_params(train_data)
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
        callbacks = []
        for layer in self.layers:
            X, finish_update = layer.begin_update(X, **kwargs)
            callbacks.append(finish_update)
        return X, self._get_finish_update(callbacks)
    
    def _get_finish_update(self, callbacks):
        def finish_update(gradient, optimizer, **kwargs):
            for callback in reversed(callbacks):
                gradient = callback(gradient, optimizer=optimizer, **kwargs)
            if optimizer is not None and self.params is not None:
                optimizer(self.params.data, self.params.gradient,
                    key=('data', self.name), **kwargs)
            return gradient
        return finish_update

    def average_params(self, averages):
        if not self.is_allocated:
            raise Exception("TODO Error")

        for layer in self.layers:
            layer.average_params(averages)
        self.params.update(averages)
        if ('data', self.name) in averages:
            self.params.data[:] = averages[('data', self.name)]


#
#    def _args2kwargs(self, names, args, **kwargs):
#        # Move positional args into the keyword args, so they can be handled
#        # via the _update_defaults machinery.
#        assert len(names) == len(args), "TODO: Error message"
#        if not args:
#            return kwargs
#        if len(args) >= 1:
#            assert 'output_shape' not in kwargs, "TODO: Error message"
#            kwargs['output_shape'] = args.pop(0)
#        if len(args) >= 1:
#            assert 'input_shape' not in kwargs, "TODO: Error message"
#            kwargs['input_shape'] = args.pop(0)
#        return kwargs
# 
