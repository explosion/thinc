# cython: profile=True
# cython: cdivision=True
# cython: infer_types = True
cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport exp, sqrt
from libc.stdlib cimport calloc, malloc, free

import numpy
from cytoolz import concat
from numpy import prod
from numpy cimport ndarray

from ..typedefs cimport weight_t


try:
    import cupy
except ImportError:
    cupy = None

try:
    import cytoolz as toolz
except ImportError:
    import toolz


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
        elif drop >= 1.0:
            return self.allocate(shape)
        coinflips = self.xp.random.uniform(0., 1., shape)
        return (coinflips >= drop) / (1.-drop)

    def allocate(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        nr_weight = numpy.prod(shape)
        return self.xp.zeros(shape, dtype='f')

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
        maxes = self.xp.max(x, axis=1)
        maxes = maxes.reshape((x.shape[0], 1))
        shifted = x - maxes
        new_x = self.xp.exp(shifted)
        new_x /= new_x.sum(axis=1).reshape((x.shape[0], 1))
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
        scale = self.xp.sqrt(6. / (W.shape[0] + W.shape[1]))
        if inplace:
            W[:] = self.xp.random.uniform(-scale, scale, W.shape)
            return W
        else:
            return self.xp.random.uniform(-scale, scale, W.shape)

    def he_normal_init(self, shape, fan_in):
        scale = self.xp.sqrt(2. / fan_in)
        return self.xp.random.normal(scale=scale, size=prod(shape)).reshape(shape)


class NumpyOps(Ops):
    xp = numpy

    def elu(self, ndarray X, inplace=True):
        cdef weight_t* data = <weight_t*>X.data
        cdef size_t size = X.size
        for i in range(size):
            if data[i] < 0:
                data[i] = exp(data[i])-1.

    def backprop_elu(self, ndarray delta_, ndarray signal_out_,
            inplace=True):
        # Backprop the ELU transformation
        # Note that this is over the function _output_, not the function
        # _input_!
        cdef size_t size = delta_.size
        cdef weight_t* delta = <weight_t*>delta_.data
        cdef const weight_t* signal_out = <const weight_t*>signal_out_.data
        for i in range(size):
            if signal_out[i] <= 0:
                delta[i] *= signal_out[i] + 1.

    def relu(self, ndarray X, inplace=True):
        cdef weight_t* data = <weight_t*>X.data
        cdef size_t size = X.size
        for i in range(size):
            if data[i] < 0:
                data[i] = 0.

    def backprop_relu(self, ndarray delta_, ndarray signal_out_, inplace=True):
        cdef size_t size = delta_.size
        cdef weight_t* delta = <weight_t*>delta_.data
        cdef const weight_t* signal_out = <const weight_t*>signal_out_.data
        for i in range(size):
            if signal_out[i] <= 0:
                delta[i] = 0.


class CupyOps(Ops):
    xp = cupy
