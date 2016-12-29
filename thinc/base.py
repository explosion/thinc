import numpy

from . import util
from .train import Trainer
from .exceptions import ShapeError


class Model(object):
    '''Model base class.'''
    name = 'model'
    device = 'cpu'
    Trainer = Trainer
    ops = None
    output_shape = None
    input_shape = None
    
    def __init__(self, *args, **kwargs):
        self.layers = []
        kwargs = self.update_defaults(*args, **kwargs)
        if self.ops is None:
            self.ops = util.get_ops(self.device)
        self.setup(*args, **kwargs)

    def update_defaults(self, *args, **kwargs):
        new_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                new_kwargs[key] = value
        return new_kwargs
    
    def setup(self, *args, **kwargs):
        pass

    def check_shape(self, x, is_batch):
        if is_batch:
            if len(x.shape) != 2:
                raise ShapeError.expected_batch(locals(), globals())
            if x.shape[1] != self.nr_in:
                dims = (x.shape[1], self.nr_in)
                raise ShapeError.dim_mismatch(dims, locals(), globals())
        else:
            if x.shape[0] != self.nr_in:
                dims = (x.shape[0], self.nr_in)
                raise ShapeError.dim_mismatch(dims, locals(), globals())

    def __call__(self, x):
        '''Predict a single x.'''
        if not self.is_initialized:
            self.set_weights(initialize=True, example=x)
        is_batch = self.is_batch(x)
        self.check_shape(x, is_batch)
        if is_batch:
            return self.predict_batch(x)
        else:
            return self.predict_one(x)

    def is_batch(self, X):
        if hasattr(X, 'shape') and len(X.shape) >= 2:
            return True
        else:
            return False
    
    def predict_one(self, x):
        X = self.ops.expand_dims(x, axis=0)
        return self.predict_batch(X)[0]

    def predict_batch(self, X):
        raise NotImplementedError
    
    def begin_training(self, train_data):
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
    
    def average_params(self, optimizer):
        pass

    def begin_update(self, X, dropout=0.0):
        raise NotImplementedError
    
    @property
    def is_initialized(self):
        return True


class Network(Model):
    '''A model that chains together other Models.'''
    name = 'mlp'

    @property
    def input_shape(self):
        return self.layers[0].input_shape

    @property
    def output_shape(self):
        return self.layers[-1].output_shape

    @property
    def nr_weight(self):
        return sum(layer.nr_weight for layer in self.layers)

    def setup(self, *layers, **kwargs):
        for i, layer in enumerate(layers):
            if isinstance(layer, Model):
                self.layers.append(layer)
            else:
                self.layers.append(layer(**kwargs))
        self.set_weights(initialize=True)
        self.set_gradient()

    def set_weights(self, example=None, data=None, initialize=True):
        if data is None:
            self.data = self.ops.allocate_pool(self.nr_weight,
                            name=(self.name, 'pool'))
        else:
            self.data = data
        for layer in self.layers:
            layer.set_weights(data=self.data, initialize=initialize)
            layer.data = None

    def set_gradient(self, data=None):
        if data is None:
            self.d_data = self.ops.allocate_pool(self.nr_weight,
                            name=(self.name, 'd_pool'))
        else:
            self.d_data = data
        for layer in self.layers:
            layer.set_gradient(data=self.d_data)
            layer.d_data = None

    def predict_batch(self, X):
        for layer in self.layers:
            X = layer.predict_batch(X)
        return X

    def begin_update(self, X, dropout=0.0):
        callbacks = []
        for layer in self.layers:
            X, finish_update = layer.begin_update(X, dropout=dropout)
            callbacks.append(finish_update)
        return X, self._get_finish_update(callbacks)

    def average_params(self, optimizer):
        for layer in self.layers:
            layer.average_params(optimizer)
        if self.data is not None and ('data', self.name) in optimizer.averages:
            self.data.data[:] = optimizer.averages[('data', self.name)]

    def _get_finish_update(self, callbacks):
        def finish_update(gradient, optimizer, **kwargs):
            for callback in reversed(callbacks):
                gradient = callback(gradient, optimizer=optimizer, **kwargs)
            if optimizer is not None and self.data is not None:
                optimizer(self.data.data, self.d_data.data,
                    key=('data', self.name), **kwargs)
            return gradient
        return finish_update
