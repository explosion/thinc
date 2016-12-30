import numpy
from cytoolz import concat


try:
    import cupy
except ImportError:
    cupy = None

try:
    import cytoolz as toolz
except ImportError:
    import toolz


class DataPool(object):
    def __init__(self, data):
        self.data = data
        self.i = 0

    def allocate(self, nr_weight):
        data = self.data[self.i : self.i + nr_weight]
        self.i += nr_weight
        return data

    def allocate_shape(self, shape):
        return self.allocate(numpy.prod(shape)).reshape(shape)


class Ops(object):
    xp = None

    def __init__(self, xp=None):
        if xp is not None:
            self.xp = xp

    def dropout(self, x, dropout, inplace=False):
        if dropout <= 0.0:
            return x, lambda func: func
        mask = self.get_dropout_mask(x.shape, dropout)
        def wrap_backprop(backprop):
            def finish_update(gradient, *args, **kwargs):
                return backprop(gradient * mask, *args, **kwargs)
            return finish_update
        if inplace:
            x *= mask
            return x, wrap_backprop
        else:
            return x * mask, wrap_backprop

    def flatten(self, X):
        return self.asarray(list(concat(X)))
 
    def unflatten(self, X, lengths):
        unflat = []
        for length in lengths:
            unflat.append(X[:length])
            X = X[length:]
        assert len(X) == 0
        assert len(unflat) == len(lengths)
        return unflat

    def get_dropout_mask(self, shape, drop):
        if drop <= 0.0:
            return None
        coinflips = self.xp.random.uniform(0., 1., shape)
        return (coinflips >= drop) / drop

    def allocate(self, shape, name=None):
        if isinstance(shape, int):
            shape = (shape,)
        nr_weight = numpy.prod(shape)
        return self.xp.zeros(shape, dtype='f')

    def allocate_pool(self, nr_weight, name=None):
        return DataPool(self.xp.zeros((nr_weight,), dtype='f'))

    def asarray(self, data, dtype='f'):
        return self.xp.asarray(data, dtype=dtype)

    def batch_dot(self, x, y):
        return self.xp.tensordot(x, y, axes=[[1], [1]])
   
    def batch_outer(self, x, y):
        return self.xp.tensordot(x, y, axes=[[0], [0]])

    def norm(self, x):
        return self.xp.sqrt((x * x).sum())

    def dot(self, x, y):
        return self.xp.dot(x, y)
    
    def affine(self, weights, bias, signal):
        return self.batch_dot(signal, weights) + bias

    def argmax(self, x, axis=-1):
        return self.xp.argmax(x, axis=axis)

    def softmax(self, x, inplace=False, axis=1):
        if x.ndim >= 3:
            raise NotImplementedError(
                "Softmax currently only supports 2d. ndim=%d" % x.ndim)
        shape = x.shape
        new_x = self.xp.zeros(shape=shape, dtype='f')
        for i in range(shape[0]):
            new_x[i] = self.xp.exp(x[i] - self.xp.max(x[i]))
            new_x[i] /= new_x[i].sum()
        if inplace:
            x[:] = new_x
            return x
        else:
            return new_x

    def expand_dims(self, a, axis=-1):
        return self.xp.expand_dims(a, axis=axis)

    def clip_low(self, x, value, inplace=False):
        if inplace:
            return self.xp.maximum(x, value, out=x)
        else:
            return self.xp.maximum(x, value)

    def take_which(self, x, which, axis=-1):
        output = self.allocate(which.shape)
        for i in range(x.shape[axis]):
            output += x[:,:,i] * (which == i)
        return output

    def xavier_uniform_init(self, W, inplace=True):
        scale = self.xp.sqrt(2. / (W.shape[0] + W.shape[1]))
        if inplace:
            W[:] = self.xp.random.uniform(-scale, scale, W.shape)
            return W
        else:
            return self.xp.random.uniform(-scale, scale, W.shape)


class NumpyOps(Ops):
    xp = numpy


class CupyOps(Ops):
    xp = cupy
