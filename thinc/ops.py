import numpy

try:
    import cupy
except ImportError:
    cupy = None


class Ops(object):
    xp = None

    def __init__(self, xp=None, reserve=0):
        if xp is not None:
            self.xp = xp
        self.data = self.xp.zeros((reserve,), dtype='f')
        self._i = 0

    def reserve(self, n):
        assert self._i == 0, "TODO Error"
        self.data = self.xp.zeros((n,), dtype='f')

    def get_dropout(self, shape, drop):
        if drop <= 0.0:
            return None
        coinflips = self.xp.random.uniform(0., 1., shape)
        return (coinflips >= drop) / drop

    def allocate(self, shape, name=None):
        if isinstance(shape, int):
            shape = (shape,)
        nr_weight = numpy.prod(shape)
        if (self._i + nr_weight) < self.data.size:
            chunk = self.data[self._i : self._i + nr_weight].reshape(shape)
            self._i += nr_weight
            return chunk
        return self.xp.zeros(shape, dtype='f')

    def asarray(self, data):
        return self.xp.asarray(data, dtype='f')

    def batch_dot(self, x, y):
        return self.xp.tensordot(x, y, axes=[[1], [1]])
   
    def batch_outer(self, x, y):
        return self.xp.tensordot(x, y, axes=[[0], [0]])

    def dot(self, x, y):
        return self.xp.dot(x, y)
    
    def affine(self, weights, bias, signal):
        return self.batch_dot(signal, weights) + bias

    def argmax(self, x, axis=-1):
        return self.xp.argmax(x, axis=axis)

    def expand_dims(self, a, axis=-1):
        return self.xp.expand_dims(a, axis=axis)

    def clip_low(self, x, value, inplace=False):
        if inplace:
            return self.xp.maximum(x, value, out=x)
        else:
            return self.xp.maximum(x, value)

    def take_which(self, x, which):
        raise NotImplementedError

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
