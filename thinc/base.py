import numpy

from . import util
from .train import Trainer


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

    def _initialize_weights(self, x):
        pass

    def check_shape(self, x, is_batch):
        if is_batch:
            if len(x.shape) != 2:
                raise ShapeError.expected_batch(locals(), globals())
            if input_BI.shape[1] != self.nr_in:
                raise ShapeError.dim_mismatch(locals(), globals())
        else:
            if input_BI.shape[1] != self.nr_in:
                raise ShapeError.dim_mismatch(locals(), globals())

    def __call__(self, x):
        '''Predict a single x.'''
        is_batch = self.is_batch(x)
        self._initialize_weights(x, is_batch)
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


class Network(Model):
    '''A model that chains together other Models.'''
    name = 'mlp'

    @property
    def input_shape(self):
        return self.layers[0].input_shape

    @property
    def output_shape(self):
        return self.layers[-1].output_shape

    def setup(self, *layers, **kwargs):
        for i, layer in enumerate(layers):
            if isinstance(layer, Model):
                self.layers.append(layer)
            else:
                self.layers.append(layer(**kwargs))
    
    def initialize_weights(self, x):
        self.params_data = self.ops.allocate_pool(self.nr_weight,
                             name=(self.name, 'pool'))
        for layer in self.layers:
            x = layer.initialize_weights(x, data=self.params_data)

    def predict_batch(self, X):
        for layer in self.layers:
            X = layer.predict_batch(X)
        return X

    def begin_update(self, X):
        callbacks = []
        for layer in self.layers:
            X, finish_update = layer.begin_update(X)
            callbacks.append(finish_update)
        return X, self._get_finish_update(backprop_callbacks)

    def _get_finish_update(self, callbacks):
        def finish_update(gradient, drop=0.0):
            for callback in reversed(callbacks):
                gradient = callback(gradient, drop=drop)
            return gradient
        return finish_update
