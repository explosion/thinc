from . import util


class Model(object):
    '''Model base class.'''
    name = 'model'
    def __init__(self, ops, *args, **kwargs):
        self.name = kwargs.get('name', self.name)
        self.ops = util.get_ops(ops)
        self.setup(*args, **kwargs)
    
    @property
    def shape(self):
        raise NotImplementedError

    @property
    def nr_out(self):
        raise NotImplementedError

    @property
    def nr_in(self):
        raise NotImplementedError

    def setup(self, components, **kwargs):
        raise NotImplementedError

    def __call__(self, x):
        '''Predict a single x.'''
        if self.is_batch(x):
            return self.predict_batch(x)
        else:
            return self.predict_one(x)

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

    def is_batch(self, X):
        raise NotImplementedError

    def predict_batch(self, X):
        raise NotImplementedError
    
    def predict_one(self, X):
        raise NotImplementedError

    def begin_update(self, X, drop=0.0):
        raise NotImplementedError


class Network(Model):
    '''A model that chains together other Models.'''
    def setup(self, *args, **kwargs):
        self.ops.reserve(self.get_nr_weight(args, **kwargs))
        self.layers = [self.make_component(i, args, **kwargs)
                       for i in range(len(args))]

    def get_nr_weight(self, components, **kwargs):
        nr_weight = 0
        for component in components:
            if hasattr(component.nr_weight):
                nr_weight += component.nr_weight
            elif isinstance(component, int):
                nr_weight += component
            elif hasattr(component, '__getitem__'):
                nr_weight += numpy.prod(component)
        return nr_weight

    def make_component(self, i, components, **kwargs):
        return components[i]

    def predict_batch(self, X):
        for layer in self.layers:
            X = layer.predict_batch(X)
        return X

    def predict_one(self, x):
        for layer in self.layers:
            X = layer.predict_one(X)
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
