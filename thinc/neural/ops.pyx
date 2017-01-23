# cython: profile=True
# cython: cdivision=True
# cython: infer_types = True
cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport exp, sqrt
from libc.stdlib cimport calloc, malloc, free
from libc.string cimport memcpy
from cymem.cymem cimport Pool

import numpy
from cytoolz import concat
from numpy import prod
from numpy cimport ndarray
from collections import Sized
cimport numpy as np

from ..typedefs cimport weight_t
from ..linalg cimport VecVec


try:
    import cupy
except ImportError:
    cupy = None

try:
    import cytoolz as toolz
except ImportError:
    import toolz


class Ops(object):
    device = 'cpu'
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

    def flatten(self, X, dtype=None):
        return self.asarray(list(concat(X)), dtype=dtype)
 
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
        return self.asarray((coinflips >= drop) / (1.-drop), dtype='float32')

    def allocate(self, shape, dtype='float32'):
        if isinstance(shape, int):
            shape = (shape,)
        nr_weight = numpy.prod(shape)
        return self.xp.zeros(shape, dtype=dtype)

    def unzip(self, data):
        X, y = zip(*data)
        return self.asarray(X), self.asarray(y)

    def asarray(self, data, dtype=None):
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

    def backprop_take(self, dX__bo, which__bo, nP):
        dX__bop = self.allocate((dX__bo.shape[0], dX__bo.shape[1], nP))
        for i in range(nP):
            dX__bop[:, :, i] += dX__bo * (which__bo == i)
        return dX__bop

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
    device = 'cpu'
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

    def maxout(self, float[:, :, ::1] py_cands):
        cdef Pool mem = Pool()
        cdef int B = py_cands.shape[0]
        cdef int O = py_cands.shape[1]
        cdef int P = py_cands.shape[2]

        which__bo = <int*>mem.alloc(B * O, sizeof(int))
        best__bo = <float*>mem.alloc(B * O, sizeof(float))
        maxout(best__bo, which__bo,
            &py_cands[0, 0, 0], B, O, P)
        cdef ndarray py_best = self.xp.ascontiguousarray(self.allocate(B * O, dtype='float32'))
        memcpy(py_best.data, best__bo, B * O * sizeof(best__bo[0]))
        cdef ndarray py_which = self.xp.ascontiguousarray(self.allocate(B * O, dtype='int32'))
        memcpy(py_which.data, which__bo, B * O * sizeof(which__bo[0]))
        return py_best.reshape((B, O)), py_which.reshape((B, O))
    
    def backprop_maxout(self, float[:, ::1] dX__bo, int[:, ::1] which__bo, int P):
        cdef Pool mem = Pool()
        cdef int B = dX__bo.shape[0]
        cdef int O = dX__bo.shape[1]

        dX__bop = <float*>mem.alloc(B * O * P, sizeof(float))
        backprop_maxout(dX__bop,
            &dX__bo[0, 0], &which__bo[0, 0], B, O, P)
        cdef ndarray py_out = self.xp.ascontiguousarray(self.allocate(B*O*P, dtype='float32'))
        memcpy(py_out.data, dX__bop, B * O * P * sizeof(dX__bop[0]))
        return py_out.reshape((B, O, P))

    def seq2col(self, float[:, ::1] seq, int nW):
        cdef int B = seq.shape[0]
        cdef int I = seq.shape[1]
        cdef Pool mem = Pool()
        cols = <float*>mem.alloc(B * I * (nW*2+1), sizeof(float))
        seq2col(cols,
            &seq[0,0], B, I, nW)
        cdef ndarray py_out = self.xp.ascontiguousarray(
            self.allocate(B*(2 * nW+1) * I, dtype='float32'))
        memcpy(py_out.data, cols, B * (2*nW+1) * I * sizeof(cols[0]))
        return py_out.reshape((B, I * 2*nW+1))


class CupyOps(Ops):
    device = 'gpu'
    xp = cupy


cdef void seq2col(float* output, const float* X, int B, int I, int nW) nogil:
    nF = nW * 2 + 1
    output += nW
    for i in range(B-nW):
        memcpy(output,
            X, I * (nW+1) * sizeof(output[0]))
        output += I * (nW+1)
        memcpy(output,
            X, I * nW * sizeof(output[0]))
        output += I * nW
        X += I
    memcpy(output,
        X, I * (nW+1) * sizeof(output[0]))


cdef void backprop_seq2col(float* dX__bi,
        const float* dY__bfi, int B, int I, int O, int nW) nogil:
    nF = nW * 2 + 1
    for i in range(B):
        # dX[-nW : nW+1] += dY
        for f in range(min(i-nW, 0), max(i+nW+1, B)):
            VecVec.add_i(dX__bi + (f * I), dY__bfi, 1., I)
        dX__bi += I


cdef void maxout(float* best__bo, int* which__bo,
        const float* cands__bop, int B, int O, int P) nogil:
    for b in range(B):
        for o in range(O):
            which__bo[0] = 0
            best__bo[0] = cands__bop[0]
            cands__bop += 1
            for p in range(1, P):
                if cands__bop[0] > best__bo[0]:
                    which__bo[0] = p
                    best__bo[0] = cands__bop[0]
                cands__bop += 1
            best__bo += 1
            which__bo += 1


cdef void backprop_maxout(float* dX__bop,
        const float* dX__bo, const int* which__bo, int B, int O, int P) nogil:
    for b in range(B):
        for o in range(O):
            dX__bop[which__bo[0]] = dX__bo[0]
            dX__bop += P
            dX__bo += 1
            which__bo += 1
