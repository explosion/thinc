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

    def get_dropout(self, shape, drop=0.0):
        if drop <= 0.0:
            return None
        coinflips = self.xp.random.uniform(0., 1., shape)
        return (coinflips >= drop) / drop

    def allocate(self, shape, name=None):
        if isinstance(shape, int):
            shape = (shape,)
        nr_weight = numpy.prod(shape)
        if (self._i + nr_weight) < self.data.size:
            chunk = self.data[self._i : self._i + nr_weight].resize(shape)
            self._i += nr_weight
            return chunk
        return self.xp.zeros(shape, dtype='f')

    def asarray(self, data):
        return self.xp.asarray(data, dtype='f')

    def batch_dot(self, weights, signal):
        return self.xp.tensordot(weights, signal, axes=[[1], [1]])
   
    def batch_outer(self, weights, signal):
        return self.xp.tensordot(weights, signal, axes=[[0], [0]])

    def dot(self, weights, signal):
        return self.xp.dot(weights, signal)
    
    def affine(self, weights, bias, signal):
        return self.batch_dot(weights, signal) + bias

    def argmax(self, x, axis=-1):
        return self.xp.argmax(x, axis=axis)

    def take_which(self, x, which):
        pass


class NumpyOps(Ops):
    xp = numpy


class CupyOps(Ops):
    xp = cupy
