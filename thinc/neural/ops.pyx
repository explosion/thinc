# cython: profile=True
# cython: cdivision=True
# cython: infer_types = True
cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport exp, sqrt, isnan
from libc.stdlib cimport srand, rand
from libc.stdlib cimport calloc, malloc, free
from libc.stdint cimport uint64_t
from libc.string cimport memcpy
from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

import numpy
from cytoolz import concat
from numpy import prod
from numpy cimport ndarray
from collections import Sized
cimport numpy as np

from ..typedefs cimport weight_t
from ..linalg cimport Mat, MatMat, MatVec, VecVec, Vec, sqrt
from murmurhash.mrmr cimport hash64


try:
    import cupy
    from chainer.functions.math.minmax import max as cupy_max
    from chainer.functions.math.minmax import argmax as cupy_argmax
    from cupy.cuda.function import Function
    from cupy.cuda.compiler import compile_with_cache
    from cupy.cuda.device import Device
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

    def dropout_sequences(self, X, dropout, inplace=False):
        if dropout <= 0.0:
            return X, lambda func: func
        masks = [self.get_dropout_mask(x.shape, dropout) for x in X]
        def wrap_backprop(backprop):
            def finish_update(gradient, *args, **kwargs):
                masked = []
                for i, mask in enumerate(masks):
                    if inplace:
                        gradient *= mask
                        masked.append(gradient)
                    else:
                        masked.append(gradient * mask)
                return backprop(masked, *args, **kwargs)
            return finish_update
        if inplace:
            for i, mask in enumerate(masks):
                X[i] *= mask
            return X, wrap_backprop
        else:
            masked = []
            for i, mask in enumerate(masks):
                masked.append(X[i] * mask)
            return masked, wrap_backprop

    def dropout(self, x, dropout, inplace=False):
        if dropout <= 0.0:
            return x, lambda func: func
        mask = self.get_dropout_mask(x.shape, dropout)
        if mask is None:
            return x, lambda func: func
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
        result = self.xp.concatenate(X)
        if dtype is not None:
            result = self.asarray(result, dtype=dtype)
        return result

    def unflatten(self, X, lengths):
        unflat = []
        for length in lengths:
            unflat.append(X[:length])
            X = X[length:]
        assert len(X) == 0
        assert len(unflat) == len(lengths)
        return unflat

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_dropout_mask(self, shape, drop):
        if drop <= 0:
            return None
        elif drop >= 1.:
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
        if dtype is not None:
            return self.xp.asarray(data, dtype=dtype)
        else:
            return self.xp.asarray(data)

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

    def update_averages(self, ema, weights, t, max_decay=0.9999):
        cdef weight_t decay = (1.0 + t) / (10.0 + t)
        if decay > max_decay:
            decay = max_decay
        ema -= (1-decay) * (ema - weights)

    def adam(self, weights, gradient, mom1, mom2, beta1, beta2, eps,
            learn_rate, mod_rate=1.):
        mom1 *= beta1
        mom2 *= beta2
        mom1 += gradient * (1.-beta1)
        mom2 += gradient * gradient * (1.-beta2)
        # Here we assume learn rate is calculated by the caller.
        # cdef weight_t a_t = learn_rate * sqrt(1-beta2**hp.t) / (1-beta1**hp.t);
        weights -= learn_rate * (mom1 / (mod_rate * self.xp.sqrt(mom2) + eps))
        gradient.fill(0)

    def clip_gradient(self, gradient, threshold):
        grad_norm = self.xp.linalg.norm(gradient)
        if grad_norm >= threshold:
            gradient *= threshold / grad_norm


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

    def relu(self, ndarray X, inplace=False):
        if inplace == False:
            return X * (X > 0)
        cdef weight_t* data = <weight_t*>X.data
        cdef size_t size = X.size
        for i in range(size):
            if data[i] < 0:
                data[i] = 0.
        return X

    def backprop_relu(self, ndarray delta_, ndarray signal_out_, inplace=False):
        if inplace == False:
            return delta_ * (signal_out_ > 0.)
        cdef size_t size = delta_.size
        cdef weight_t* delta = <weight_t*>delta_.data
        cdef const weight_t* signal_out = <const weight_t*>signal_out_.data
        for i in range(size):
            if signal_out[i] <= 0:
                delta[i] = 0.
        return delta_

    def maxout(self, float[:, :, ::1] py_cands):
        cdef Pool mem = Pool()
        cdef int B = py_cands.shape[0]
        cdef int O = py_cands.shape[1]
        cdef int P = py_cands.shape[2]

        which__bo = <int*>mem.alloc(B * O, sizeof(int))
        best__bo = <float*>mem.alloc(B * O, sizeof(float))
        cpu_maxout(best__bo, which__bo,
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
        cpu_backprop_maxout(dX__bop,
            &dX__bo[0, 0], &which__bo[0, 0], B, O, P)
        cdef ndarray py_out = self.xp.ascontiguousarray(self.allocate(B*O*P, dtype='float32'))
        memcpy(py_out.data, dX__bop, B * O * P * sizeof(dX__bop[0]))
        return py_out.reshape((B, O, P))

    def seq2col(self, float[:, ::1] seq, int nW):
        '''Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1)) sequence.
        The new sequence is constructed by concatenating nW preceding and succeeding
        vectors onto each column in the sequence, to extract a window of features.
        '''
        cdef int B = seq.shape[0]
        cdef int I = seq.shape[1]
        cdef Pool mem = Pool()
        cols = <float*>mem.alloc(B * I * (nW*2+1), sizeof(float))
        seq2col(cols,
            &seq[0,0], B, I, nW)
        cdef ndarray py_out = self.xp.ascontiguousarray(
            self.allocate(B*(2 * nW+1) * I, dtype='float32'))
        memcpy(py_out.data, cols, B * (2*nW+1) * I * sizeof(cols[0]))
        return py_out.reshape((B, I * (2*nW+1)))

    def backprop_seq2col(self, float[:, ::1] dY, int nW):
        cdef int B = dY.shape[0]
        cdef int nF = nW*2+1
        cdef int I = dY.shape[1] / nF
        cdef Pool mem = Pool()
        dX = <float*>mem.alloc(B * I, sizeof(float))
        backprop_seq2col(dX, &dY[0,0], B, I, nW)
        cdef ndarray py_out = self.xp.ascontiguousarray(
            self.allocate(B * I, dtype='float32'))
        memcpy(py_out.data, dX, B * I * sizeof(dX[0]))
        return py_out.reshape((B, I))

    def remap_ids(self, PreshMap mapping, uint64_t[::1] ids_mv, uint64_t value=0):
        cdef uint64_t* ids = &ids_mv[0]
        cdef ndarray[uint64_t] output_arr = self.allocate(len(ids_mv), dtype='uint64')
        output = <uint64_t*>output_arr.data
        cdef uint64_t key = 0
        for i in range(ids_mv.shape[0]):
            if ids[i] == 0:
                output[i] = 0
            else:
                mapped = <uint64_t>mapping.get(ids[i])
                if mapped != 0:
                    output[i] = mapped
                else:
                    output[i] = value
                    if value != 0:
                        mapping.set(ids[i], <void*>value)
                        value += 1
        return output_arr

    def increment_slices(self, ndarray contig_array, ndarray _to_add, _starts):
        cdef ndarray contig_to_add = self.xp.ascontiguousarray(_to_add, dtype='float32')
        cdef ndarray contig_starts = self.xp.ascontiguousarray(_starts, dtype='int32')

        cdef const float* to_add = <const weight_t*>contig_to_add.data
        cdef float* whole_array = <weight_t*>contig_array.data
        cdef const int* starts = <const int*>contig_starts.data
        cdef int n_slice = len(_starts)
        cdef int length = _to_add.size
        cdef int stride = length / _to_add.shape[0]
        for start in starts[:n_slice]:
            workon = &whole_array[start * stride]
            for i in range(length):
                workon[i] += to_add[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def hash(self, uint64_t[::1] ids, uint64_t seed):
        cdef ndarray[uint64_t, ndim=1] keys = self.allocate(ids.size, dtype='uint64')
        cdef int i
        for i in range(ids.size):
            keys[i] = hash64(&ids[i], sizeof(uint64_t), seed)
        return keys

    def mean_pool(self, float[:, ::1] X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        cdef Pool mem = Pool()
        means = <float*>mem.alloc(B * O, sizeof(float))

        cpu_mean_pool(means,
            &X[0, 0], &lengths[0], B, T, O)
        return cpu_floats_ptr2array(means, (B, O))

    def backprop_mean_pool(self, float[:, ::1] d_means, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_means.shape[1]
        cdef int T = 0
        for length in lengths[:B]:
            T += length
        cdef Pool mem = Pool()
        dX = <float*>mem.alloc(T * O, sizeof(float))

        cpu_backprop_mean_pool(dX,
            &d_means[0,0], &lengths[0], B, T, O)

        return cpu_floats_ptr2array(dX, (T, O))

    def max_pool(self, float[:, ::1] X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        cdef Pool mem = Pool()
        maxes = <float*>mem.alloc(B * O, sizeof(float))
        which = <int*>mem.alloc(B * O, sizeof(int))

        cpu_max_pool(maxes, which,
            &X[0, 0], &lengths[0], B, T, O)

        cdef ndarray py_best = cpu_floats_ptr2array(maxes, (B, O))
        cdef ndarray py_which = cpu_ints_ptr2array(which, (B, O))
        return py_best, py_which

    def backprop_max_pool(self, float[:, ::1] d_maxes,
            int[:, ::1] which, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_maxes.shape[1]
        cdef int T = 0
        for length in lengths[:B]:
            T += length
        cdef Pool mem = Pool()
        dX = <float*>mem.alloc(T * O, sizeof(float))

        cpu_backprop_max_pool(dX,
            &d_maxes[0,0], &which[0, 0], &lengths[0], B, T, O)

        return cpu_floats_ptr2array(dX, (T, O))

    #def adam(self, float[::1] weights, float[::1] gradient, float[::1] mom1,
    #        float[::1] mom2, float beta1, float beta2, float eps,
    #        float learn_rate, float mod_rate=1.):
    #    _adam(&weights[0], &gradient[0], &mom1[0], &mom2[0],
    #        weights.shape[0], beta1, beta2, eps, learn_rate, mod_rate)


@cython.cdivision(True)
cdef void _adam(
    weight_t* weights, weight_t* gradient, weight_t* mom1, weight_t* mom2, 
        int nr_weight, weight_t beta1, weight_t beta2, weight_t eps,
        weight_t learn_rate, weight_t mod_rate) nogil:
    Vec.mul_i(mom1,
        beta1, nr_weight)
    VecVec.add_i(mom1,
        gradient, 1-beta1, nr_weight)

    for i in range(nr_weight):
        gradient[i] *= gradient[i] * (1-beta2)
    Vec.mul_i(mom2,
        beta2, nr_weight)
    VecVec.add_i(mom2, gradient, 1.0, nr_weight)
    #for i in range(nr_weight):
    #    mom2[i] = (beta2 * mom2[i]) + ((1-beta2) * gradient[i] * gradient[i])
    # Here we assume this is calculated by the caller.
    #cdef weight_t a_t = learn_rate * sqrt(1-beta2**hp.t) / (1-beta1**hp.t)
    for i in range(nr_weight):
        weights[i] -= learn_rate * (mom1[i] / (mod_rate * sqrt(mom2[i]) + eps))
    memset(gradient, 0, sizeof(gradient[0]) * nr_weight)


@cython.cdivision(True)
cdef void cpu_update_averages(weight_t* ema,
        const weight_t* weights, int nr_weight, weight_t t, weight_t max_decay) nogil:
    cdef weight_t decay = (1.0 + t) / (10.0 + t)
    if decay > max_decay:
        decay = max_decay
    for i in range(nr_weight):
        ema[i] -= (1-decay) * (ema[i] - weights[i])


class CupyOps(Ops):
    device = 'gpu'
    xp = cupy

    def maxout(self, X):
        amax = cupy_max(X, axis=-1).data
        argmax = cupy_argmax(X, axis=-1).data
        return amax, cupy.asarray(argmax, dtype='int32')

    def backprop_maxout(self, dX__bo, which__bo, int P):
        dX__bop = gpu_backprop_maxout(
            dX__bo.ravel(), which__bo.ravel(), P, size=dX__bo.size * P)
        return dX__bop.reshape((dX__bo.shape[0], dX__bo.shape[1], P))

    def relu(self, X, inplace=False):
        if not inplace:
            return X * (X > 0)
        else:
            X *= (X > 0)
            return X

    def backprop_relu(self, delta_, signal_out, inplace=False):
        if not inplace:
            return delta_ * (signal_out > 0)
        delta_ *= (signal_out > 0)
        return delta_

    def clip_gradient(self, gradient, threshold):
        grad_norm = self.xp.linalg.norm(gradient)
        if grad_norm >= threshold:
            gradient *= threshold / grad_norm

    def seq2col(self, seq, int nW):
        '''Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1)) sequence.
        The new sequence is constructed by concatenating nW preceding and succeeding
        vectors onto each column in the sequence, to extract a window of features.
        '''
        cdef int B = seq.shape[0]
        cdef int I = seq.shape[1]
        cols = self.allocate((B, (nW*2+1), I))
        cols[:-1, 0] = seq[1:]
        cols[:, 1] = seq
        cols[1:, 2] = seq[:-1]
        return cols.reshape((B, I * (2*nW+1)))

    def backprop_seq2col(self, dY, int nW):
        cdef int nF = nW*2+1
        cdef int B = dY.shape[0]
        cdef int I = dY.shape[1] / nF
        assert nF == 3, "TODO: Support variable window size" 
        # Having trouble getting the kernel to work...
        dX = self.allocate((B, I))
        dY = dY.reshape((B, nF, I))
        dX[1:] += dY[:-1, 0]
        dX += dY[:, nW]
        dX[:-1] += dY[1:, 2]
        return dX


class RawKernel(object):
    def __init__(self, name, src):
        if cupy is None:
            return None
        # TODO: Some setup is performed when a function is launched that I haven't
        # identified yet. Without this setup, loading the function fails...
        _ = cupy.ElementwiseKernel('float32 x, float32 y', 'float32 z',
            'z = (x - y) * (x - y)', 'squared_diff')
        with Device(0):
            _(cupy.ones((5,), dtype='float32'), cupy.ones((5,), dtype='float32'))
            self.func = Function(compile_with_cache(src), name)

    def __call__(self, *args, **kwargs):
        # TODO: Not convinced I'm doing this right...
        grid = kwargs.get('grid', (args[0].shape[0],))
        block = kwargs.get('block', (1,)) 
        return self.func(grid, block, args)


# TODO: Currently broken
gpu_seq2col = RawKernel('gpu_seq2col', '''
extern "C" __global__
void gpu_seq2col(float* output, const float* X, int B, int I, int nW)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= B)
        return;
    int nF = nW * 2 + 1;

    output += i * I * nF;
    X += i * I;
    if (i == (B-nW)) {
        for (int j = 0; j < (I*nW); ++j)
            output[j] = X[j];
        return;
    }
    for (int j = 0; j < (I*(nW+1)); ++j)
        output[j] = X[j];
    output += I * (nW+1);
    for (int j = 0; j < (I*nW); ++j)
        output[j] = X[j];
}
''')

# TODO: This doesn't work yet. Segfaults.
gpu_backprop_seq2col = RawKernel('gpu_backprop_seq2col', '''
extern "C" __global__
void gpu_backprop_seq2col(float* d_seqs, const float* d_cols, int B, int I, int nW)
{
    return;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int nF = nW * 2 + 1;
 
    if ((i+nW) >= B) return;
    
    float* d_seq = &d_seqs[i * I];
    // Here's what we're doing, if we had 2d indexing.
    // d_seq += d_cols[i-2, 4]
    // d_seq += d_cols[i-1, 3]
    // d_seq += d_cols[i, 2]
    // d_seq += d_cols[i+1, 1]
    // d_seq += d_cols[i+2, 0]
 
    const float* col_row = &d_cols[i * I * nF];
    //for (int f = -nW; f < (nW+1); ++f) {
    //    if (B > (i+f) && (i+f) >= 0) {
    //        const float* feat = col_row + (f * I) + (f+nW) * I;
    //        for (int j = 0; j < I; ++j)
    //            d_seq[j] += feat[j];
    //    }
    //}
}
''')


cdef void seq2col(float* output, const float* X, int B, int I, int nW) nogil:
    nF = nW * 2 + 1
    output += nW * I
    for i in range(B-nW):
        memcpy(output,
            X, I * (nW+1) * sizeof(output[0]))
        output += I * (nW+1)
        memcpy(output,
            X, I * nW * sizeof(output[0]))
        output += I * nW
        X += I
    memcpy(output,
        X, I * nW * sizeof(output[0]))


cdef void backprop_seq2col(float* d_seqs,
        const float* d_cols, int B, int I, int nW) nogil:
    # Here's what we're doing, if we had 2d indexing.
    #for i in range(B):
    #    d_seq[i] += d_cols[i-2, 4]
    #    d_seq[i] += d_cols[i-1, 3]
    #    d_seq[i] += d_cols[i+2, 0]
    #    d_seq[i] += d_cols[i+1, 1]
    #    d_seq[i] += d_cols[i, 2]
    nF = nW * 2 + 1
    for i in range(B):
        seq_row = &d_seqs[i * I]
        col_row = &d_cols[i * I * nF]
        for f in range(-nW, nW+1):
            if B > (i+f) >= 0:
                feat = col_row + (f * I)
                VecVec.add_i(seq_row, &feat[(f+nW) * I], 1., I)


cdef void cpu_maxout(float* best__bo, int* which__bo,
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


cdef void cpu_backprop_maxout(float* dX__bop,
        const float* dX__bo, const int* which__bo, int B, int O, int P) nogil:
    for b in range(B):
        for o in range(O):
            dX__bop[which__bo[0]] = dX__bo[0]
            dX__bop += P
            dX__bo += 1
            which__bo += 1


# Here we broadcast over the longest dimension (dX) and compute indexes
# for the narrower dimensions.
if cupy is not None:
    gpu_backprop_maxout = cupy.ElementwiseKernel(
        'raw float32 best, raw int32 which, raw int32 P',
        'float32 dX',
        'dX = (which[i/P] == i%P) ? best[i/P] : 0',
        'bp_maxout')
    # 't2b' is a mapping from the T dimension (i.e. lengths.sum()) to
    # the B dimension. It tells you which sequence the index is in.
    gpu_backprop_max_pool = cupy.ElementwiseKernel(
        ('raw float32 d_best, raw int32 which,'
         'raw int32 lengths, raw int32 t2b, raw int32 O'),
        'float32 dX',
        '''
        dX = (which[t2b[i/O]] == i % O) ? d_best[t2b[i/O]] : 0',
        ''',
        'bp_maxpool'
    )


def cpu_clip_gradient(weight_t[::1] gradient, weight_t threshold):
    grad_norm = Vec.norm(&gradient[0], gradient.shape[0])
    if grad_norm >= threshold:
        Vec.mul_i(&gradient[0], threshold / grad_norm, gradient.shape[0])


def add_gradient_noise(float[::1] gradient, weight_t noise_level,
        weight_t timestep):
    variance = noise_level / ((1 + timestep) ** 0.55)
    if variance >= 0.000001:
        gradient += numpy.asarray(
                       numpy.random.normal(scale=variance, loc=0., size=len(gradient)),
                       dtype='float32')



cdef cpu_floats_ptr2array(const float* ptr, shape):
    cdef ndarray py_out = numpy.zeros(shape, dtype='float32')
    cdef int N = numpy.prod(shape)
    memcpy(py_out.data, ptr, N * sizeof(ptr[0]))
    return py_out


cdef cpu_ints_ptr2array(const int* ptr, shape):
    cdef ndarray py_out = numpy.zeros(shape, dtype='int32')
    cdef int N = numpy.prod(shape)
    memcpy(py_out.data, ptr, N * sizeof(ptr[0]))
    return py_out


cdef void cpu_mean_pool(float* means__bo,
        const float* X__to, const int* lengths__b,
        int B, int T, int O) nogil:
    '''Compute means of a batch of concatenated sequences, using the lengths.'''
    cdef float scale = 0.
    for length in lengths__b[:B]:
        scale = 1. / length
        for _ in range(length):
            VecVec.add_i(means__bo,
                X__to, scale, O)
            X__to += O
        means__bo += O


cdef void cpu_backprop_mean_pool(float* dX__to,
        const float* d_means__bo, const int* lengths__b,
        int B, int T, int O) nogil:
    cdef float scale = 0.
    for length in lengths__b[:B]:
        scale = 1./ length
        for _ in range(length):
            VecVec.add_i(dX__to,
                d_means__bo, scale, O)
            dX__to += O
        d_means__bo += O


cdef void cpu_max_pool(float* maxes__bo, int* which__bo,
        const float* X__to, const int* lengths__b,
        int B, int T, int O) nogil:
    '''Compute maxes of a batch of concatenated sequences, using the lengths.'''
    cdef float scale = 0.
    for length in lengths__b[:B]:
        memcpy(maxes__bo, X__to, O * sizeof(maxes__bo[0]))
        memset(which__bo, 0, O * sizeof(which__bo[0]))
        X__to += O
        for i in range(1, length):
            for j in range(O):
                if X__to[j] > maxes__bo[j]:
                    maxes__bo[j] = X__to[j]
                    which__bo[j] = i
            X__to += O
        maxes__bo += O
        which__bo += O


cdef void cpu_backprop_max_pool(float* dX__to,
        const float* d_maxes__bo, const int* which__bo, const int* lengths__b,
        int B, int T, int O) nogil:
    cdef int length, i, j
    for length in lengths__b[:B]:
        for i in range(length):
            for j in range(O):
                if which__bo[j] == i:
                    dX__to[j] += d_maxes__bo[j]
            dX__to += O
        d_maxes__bo += O
        which__bo += O


