# cython: cdivision=True, infer_types=True, profile=True
cimport cython
cimport cython.parallel
from libc.string cimport memcpy, memset
from libc.stdlib cimport srand, rand
from libc.stdlib cimport calloc, malloc, free
from libc.stdint cimport uint32_t, uint64_t
from libc.string cimport memcpy
from libc.math cimport isnan
from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

import numpy
from numpy import prod
from numpy cimport ndarray
from collections import Sized
cimport numpy as np

from ._aligned_alloc import zeros_aligned
from ..typedefs cimport weight_t
from .util import copy_array, get_array_module
from ..linalg cimport VecVec, Vec

from murmurhash.mrmr cimport hash64, hash128_x86, hash128_x64
from ..compat import integer_types

cimport blis
cimport blis.cy
import blis.py


cdef extern from "math.h":
    float sqrtf(float x) nogil
    float expf(float x) nogil
    float tanhf(float x) nogil


try:
    import cupy
    import cupy.cuda
    from cupy.cuda.function import Function
    from cupy.cuda.compiler import compile_with_cache
    # This is important -- without setting these global pools, we're
    # *very* slow -- 5x slower on mnist.
    memory_pool = cupy.cuda.MemoryPool()
    cupy.cuda.set_allocator(memory_pool.malloc)
    pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
    cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)
except ImportError:
    cupy = None


try:
    import thinc_gpu_ops as gpu_ops
except ImportError:
    pass


class Ops(object):
    device = 'cpu'
    xp = None

    def __init__(self, xp=None):
        if xp is not None:
            self.xp = xp

    def dropout_sequences(self, X, dropout, inplace=False):
        if dropout is None or dropout <= 0.0:
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
        if dropout is None or dropout <= 0.0:
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

    def flatten(self, X, dtype=None, pad=0):
        if X is None or len(X) == 0:
            return self.allocate((0,), dtype=dtype or 'f')
        X = [x for x in X if x.size != 0]
        xp = get_array_module(X[0])
        if int(pad) >= 1:
            padded = []
            for x in X:
                padded.append(
                    xp.zeros((pad,) + x.shape[1:], dtype=x.dtype))
                padded.append(x)
            padded.append(
                xp.zeros((pad,) + x.shape[1:], dtype=x.dtype))
            X = padded
        result = xp.concatenate(X)
        if dtype is not None:
            result = xp.asarray(result, dtype=dtype)
        return result

    def unflatten(self, X, lengths, pad=0):
        unflat = []
        pad = int(pad)
        for length in lengths:
            length = int(length)
            if pad >= 1 and length != 0:
                X = X[pad:]
            unflat.append(X[:length])
            X = X[length:]
        if pad >= 1 and length != 0:
            X = X[pad:]
        assert len(X) == 0
        assert len(unflat) == len(lengths)
        return unflat

    def square_sequences(self, seqs):
        '''Sort a batch of sequence by decreasing length, pad, and transpose
        so that the outer dimension is the timestep. Return the padded batch,
        along with an array indicating the actual length at each step, and a callback
        to reverse the transformation.
        '''
        lengths_indices = [(len(seq), i) for i, seq in enumerate(seqs)]
        lengths_indices.sort(reverse=True)
        indices = [i for length, i in lengths_indices]
        lengths = [length for length, i in lengths_indices]
        nB = len(seqs)
        nS = max([len(seq) for seq in seqs])
        arr = self.allocate((nB, nS) + seqs[0].shape[1:], dtype=seqs[0].dtype)
        for arr_i, (length, seqs_i) in enumerate(lengths_indices):
            arr[arr_i, :length] = self.asarray(seqs[seqs_i])
        extra_dims = tuple(range(2, len(arr.shape)))
        arr = self.xp.ascontiguousarray(arr.transpose((1, 0) + extra_dims))
        # Build a lookup table so we can find how big the batch is at point t.
        batch_size_at_t = self.allocate((nS,), dtype='i')
        batch_size_at_t += 1
        i = len(lengths)
        for t in range(nS):
            if t == lengths[i-1]:
                i -= 1
                if i == 0:
                    break
            batch_size_at_t[t] = i
        def unpad(padded):
            unpadded = [None] * len(lengths)
            padded = self.xp.ascontiguousarray(padded.transpose((1, 0) + extra_dims))
            for i in range(padded.shape[0]):
                unpadded[indices[i]] = padded[i, :lengths[i]]
            return unpadded
        return arr, batch_size_at_t, unpad

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_dropout_mask(self, shape, drop):
        if drop is None or drop <= 0:
            return None
        elif drop >= 1.:
            return self.allocate(shape)
        coinflips = self.xp.random.uniform(0., 1., shape)
        mask = (coinflips >= drop) / (1.-drop)
        return self.asarray(mask, dtype='float32')

    def allocate(self, shape, dtype='float32'):
        if isinstance(shape, integer_types):
            shape = (shape,)
        nr_weight = numpy.prod(shape)
        return self.xp.zeros(shape, dtype=dtype)

    def unzip(self, data):
        X, y = zip(*data)
        return self.asarray(X), self.asarray(y)

    def asarray(self, data, dtype=None):
        if isinstance(data, self.xp.ndarray):
            if dtype is not None:
                return self.xp.asarray(data, dtype=dtype)
            else:
                return self.xp.asarray(data)
        elif hasattr(data, 'numpy'):
            # Handles PyTorch Tensor
            return data.numpy()
        elif dtype is not None:
            return self.xp.array(data, dtype=dtype)
        else:
            return self.xp.array(data)

    def batch_dot(self, x, y, transpose=False):
        # TODO: Fix this confusing inversion =/
        if not transpose:
            return self.xp.dot(x, y.T)
        else:
            return self.xp.dot(x, y)

    def add_batch_outer(self, output, x, y):
        # TODO: Deprecate this
        output += self.xp.tensordot(x, y, axes=[[0], [0]])

    def norm(self, x):
        return self.xp.sqrt((x * x).sum())

    def dot(self, x, y):
        # TODO: Deprecate this
        return self.xp.dot(x, y)

    def affine(self, weights, bias, signal):
        return self.gemm(signal, weights, trans2=True) + bias

    def add_sum(self, out, to_sum):
        out += to_sum.sum(axis=0)

    def argmax(self, x, axis=-1):
        return self.xp.argmax(x, axis=axis)
    
    def sigmoid(self, X):
        return 1./(1. + self.xp.exp(-X))

    def dsigmoid(self, y):
        return y*(1-y)

    def dtanh(self, y):
        return 1-y**2

    def softmax(self, x, inplace=False, axis=-1):
        if x.ndim >= 3:
            raise NotImplementedError(
                "Softmax currently only supports 2d. ndim=%d" % x.ndim)
        shape = x.shape
        maxes = self.xp.max(x, axis=axis, keepdims=True)
        shifted = x - maxes
        new_x = self.xp.exp(shifted)
        new_x /= new_x.sum(axis=axis, keepdims=True)
        if inplace:
            copy_array(x, new_x)
            return x
        else:
            return new_x

    def softmax_sequences(self, Xs, lengths, inplace=False, axis=-1):
        if Xs.ndim >= 3:
            raise NotImplementedError(
                "Softmax currently only supports 2d. ndim=%d" % Xs.ndim)
        # This loses almost no fidelity, and helps the numerical stability.
        Xs = self.xp.clip(Xs, -20., 20.)
        new_x = self.xp.exp(Xs)
        summed = self.backprop_sum_pool(self.sum_pool(new_x, lengths), lengths)
        new_x /= summed
        if inplace:
            copy_array(Xs, new_x)
            return Xs
        else:
            return new_x

    def backprop_softmax_sequences(self, dy, y, lengths):
        dx = y * dy
        sumdx = self.backprop_sum_pool(self.sum_pool(dx, lengths), lengths)
        dx -= y * sumdx
        return dx

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

    def lstm(self, output, cells, act_pieces, prev):
        hf, hi, ho, hc = act_pieces
        hf[:] = self.sigmoid(hf)
        hi[:] = self.sigmoid(hi)
        ho[:] = self.sigmoid(ho)
        hc[:] = self.xp.tanh(hc)

        cells[:] = hf * prev + hi * hc
        output[:] = self.xp.tanh(cells) * ho

    def backprop_lstm(self, d_cells, d_prev, d_gate_pieces,
            d_output, gate_pieces, cells, prev):
        hf, hi, ho, hc = gate_pieces
        dhf, dhi, dho, dhc = d_gate_pieces

        ct = self.xp.tanh(cells)

        # Gradient for ho and c in h = sigmoid(ho) * tanh(c)
        dho[:] = ct * d_output * self.dsigmoid(ho)
        dc = ho * d_output * self.dtanh(ct)
        dc += d_cells # Carry gradient from previous step

        # Gradient for hf, hi, hc, prev[i]
        # in c = sigmoid(hf) * prev[i] + sigmoid(hi) * tanh(hc)
        dhf[:] = self.dsigmoid(hf) * dc * prev
        dhi[:] = self.dsigmoid(hi) * dc * hc
        dhc[:] = self.dtanh(hc)    * dc * hi

        d_prev[:] = dc * hf
        copy_array(d_cells, dc)

    def xavier_uniform_init(self, W, inplace=True):
        if (W**2).sum() != 0.:
            return W
        scale = self.xp.sqrt(6. / (W.shape[0] + W.shape[1]))
        if inplace:
            copy_array(W, self.xp.random.uniform(-scale, scale, W.shape))
            return W
        else:
            return self.xp.random.uniform(-scale, scale, W.shape)
    
    def normal_init(self, W, fan_in, inplace=True):
        if (W**2).sum() != 0.:
            return W
        scale = self.xp.sqrt(1. / fan_in)
        inits = self.xp.random.normal(scale=scale, size=int(prod(W.shape)))
        inits = inits.reshape(W.shape)
        if inplace:
            copy_array(W, inits)
            return W
        else:
            return inits

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
        xp = get_array_module(gradient)
        grad_norm = xp.linalg.norm(gradient)
        if grad_norm >= threshold:
            gradient *= threshold / grad_norm

    def logloss(self, y_true, y_pred):
        log_yp = self.xp.log(y_pred + 1e-8)
        loss = (y_true * log_yp) + (1-y_true) * self.xp.log((1-y_pred)+1e-8)
        return -loss


class NumpyOps(Ops):
    device = 'cpu'
    xp = numpy

    def allocate(self, shape, dtype='float32'):
        if isinstance(shape, integer_types):
            shape = (shape,)
        return self.xp.zeros(shape, dtype=dtype)

    def inplace_add(self, np.ndarray x, np.ndarray y, float scale=1.0):
        VecVec.add_i(<float*>x.data,
            <float*>y.data, scale, x.shape[0])

    def gemm(self, float[:, ::1] x, float[:, ::1] y, trans1=False, trans2=False,
             out=None):
        cdef int m
        if trans1:
            m = x.shape[1]
        else:
            m = x.shape[0]
        cdef int n
        if trans2: 
            n = y.shape[0]
        else:
            n = y.shape[1]
        cdef np.ndarray out_array
        if out is None:
            out_array = self.allocate((m, n))
        else:
            out_array = self.xp.asarray(out)
        assert out_array.shape[0] == m
        assert out_array.shape[1] == n
        blis.py.gemm(x, y, out=out_array, trans1=trans1, trans2=trans2)
        return out_array

    def affine(self, weights, bias, signal):
        dotted = self.gemm(signal, weights, trans2=True)
        dotted += bias
        return dotted

    def elu(self, ndarray X, inplace=True):
        cdef weight_t* data = <weight_t*>X.data
        cdef size_t size = X.size
        for i in range(size):
            if data[i] < 0:
                data[i] = expf(data[i])-1.

    def selu(self, ndarray X, inplace=True):
        cdef weight_t* data = <weight_t*>X.data
        cdef size_t size = X.size
        cdef float scale = 1.0507009873554805
        cdef float alpha = 1.6732632423543772
        for i in range(size):
            if data[i] < 0:
                data[i] = alpha * (expf(data[i])-1.)
            data[i] *= scale

    def backprop_selu(self, ndarray delta_, ndarray signal_in_,
            inplace=True):
        # Backprop the SELU transformation
        cdef size_t size = delta_.size
        cdef weight_t* delta = <weight_t*>delta_.data
        cdef const weight_t* signal_in = <const weight_t*>signal_in_.data
        cdef float scale = 1.0507009873554805
        cdef float alpha = 1.6732632423543772

        for i in range(size):
            delta[i] *= scale
            if signal_in[i] <= 0:
                delta[i] *= alpha * expf(signal_in[i])

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
        cdef np.ndarray out = X if inplace else X.copy()
        cdef weight_t* data = <weight_t*>out.data
        cdef size_t size = out.size
        for i in range(size):
            if data[i] < 0:
                data[i] = 0.
        return out

    def backprop_relu(self, ndarray dY, ndarray Y, inplace=False):
        cdef np.ndarray dX = dY if inplace else dY.copy()
        cdef size_t size = dX.size
        cdef weight_t* dX_ptr = <weight_t*>dX.data
        cdef const weight_t* Y_ptr = <const weight_t*>Y.data
        for i in range(size):
            if Y_ptr[i] <= 0:
                dX_ptr[i] = 0.
        return dX

    def maxout(self, float[:, :, ::1] py_cands):
        cdef Pool mem = Pool()
        cdef int B = py_cands.shape[0]
        cdef int O = py_cands.shape[1]
        cdef int P = py_cands.shape[2]

        cdef ndarray best = numpy.zeros((B, O), dtype='float32', order='C')
        cdef ndarray which = numpy.zeros((B, O), dtype='int32', order='C')
        cpu_maxout(<float*>best.data, <int*>which.data,
            &py_cands[0, 0, 0], B, O, P)
        return best, which

    def backprop_maxout(self, float[:, ::1] dX__bo, int[:, ::1] which__bo, int P):
        cdef int B = dX__bo.shape[0]
        cdef int O = dX__bo.shape[1]

        cdef np.ndarray dX__bop = numpy.zeros((B, O, P), dtype='float32')
        cpu_backprop_maxout(<float*>dX__bop.data,
            &dX__bo[0, 0], &which__bo[0, 0], B, O, P)
        return dX__bop

    #def lstm(self, float[:, ::1] output, float[:, ::1] cells,
    #        float[:, ::1] gates, float[:, ::1] prev):
    #    cpu_lstm_gates_fwd(&output[0, 0], &cells[0, 0],
    #        &gates[0, 0], &prev[0, 0], cells.shape[0], cells.shape[1])
    #    return output

    #def backprop_lstm(self, float[:, ::1] d_cells, float[:, ::1] d_prev,
    #        float[:, ::1] d_gates, float[:, ::1] d_output,
    #        float[:, ::1] gates, float[:, ::1] cells, float[:, ::1] prev):
    #    cpu_lstm_gates_bwd(&d_cells[0, 0], &d_prev[0, 0], &d_gates[0, 0],
    #        &d_output[0, 0], &gates[0, 0], &cells[0, 0], &prev[0, 0],
    #        cells.shape[0], cells.shape[1])

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
    def hash(self, uint64_t[::1] ids, uint32_t seed):
        '''Hash a sequence of 64-bit keys into a table with 4 32-bit keys'''
        # Written to mirror the GPU implementation
        cdef ndarray[uint32_t, ndim=2] keys = self.allocate((ids.shape[0], 4), dtype='uint32')
        cdef int i, j
        cdef unsigned char entropy[16] # 128/8=16
        cdef size_t n_items = len(ids)
        cdef size_t in_size = sizeof(uint64_t)
        src = <unsigned char*>&ids[0]
        dest = <unsigned char*>keys.data
        for i in range(n_items):
            hash128_x64(<void*>src, in_size, seed, entropy)
            for j in range(16):
                dest[j] = entropy[j]
            src += in_size
            dest += 16
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

    def sum_pool(self, float[:, ::1] X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        cdef Pool mem = Pool()
        sums = <float*>mem.alloc(B * O, sizeof(float))

        cpu_sum_pool(sums,
            &X[0, 0], &lengths[0], B, T, O)
        return cpu_floats_ptr2array(sums, (B, O))

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

    def backprop_sum_pool(self, float[:, ::1] d_sums, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_sums.shape[1]
        cdef int T = 0
        for length in lengths[:B]:
            T += length
        cdef Pool mem = Pool()
        dX = <float*>mem.alloc(T * O, sizeof(float))

        cpu_backprop_sum_pool(dX,
            &d_sums[0,0], &lengths[0], B, T, O)
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

    def add_sum(self, np.ndarray out, np.ndarray to_sum):
        VecVec.batch_add_i(<float*>out.data,
            <const float*>to_sum.data, 1., to_sum.shape[1], to_sum.shape[0])

    def scatter_add(self, np.ndarray out, np.ndarray ids, np.ndarray inputs):
        if out.dtype == 'float32' \
        and ids.dtype == 'int32' \
        and inputs.dtype == 'float32' \
        and out.flags.c_contiguous \
        and ids.flags.c_contiguous \
        and inputs.flags.c_contiguous \
        and ids.ndim == 1 \
        and out.ndim == 2 \
        and inputs.ndim == 2 \
        and inputs.shape[0] == ids.shape[0] \
        and inputs.shape[1] == out.shape[1]:
            cpu_scatter_add(<float*>out.data,
                <int*>ids.data, <float*>inputs.data,
                ids.shape[0], out.shape[1])
        else:
            self.xp.add.at(out, ids, inputs)
 
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def adam(self, float[::1] weights, float[::1] gradient, float[::1] mom1,
            float[::1] mom2, float beta1, float beta2, float eps,
            float learn_rate, float mod_rate=1.):
        _adam_momentum(&gradient[0], &mom1[0], &mom2[0],
            weights.shape[0], beta1, beta2, eps, learn_rate)
        VecVec.add_i(&weights[0],
            &gradient[0], -learn_rate, weights.shape[0])
        memset(&gradient[0], 0, gradient.size * sizeof(float))

    def ngrams(self, int n, uint64_t[::1] keys_):
        keys = <uint64_t*>&keys_[0]
        cdef np.ndarray output_ = self.allocate((keys_.shape[0]-n,), dtype='uint64')
        output = <uint64_t*>output_.data
        for i in range(keys_.shape[0]-n):
            output[i] = hash64(&keys[i], n*sizeof(keys[0]), 0)
        return output_


cdef void cpu_scatter_add(float* dest,
        const int* indices, const float* src,
        int nr_id, int nr_col) nogil:
    cdef int i
    for i in range(nr_id):
        id_ = indices[i]
        if id_ >= 0:
            VecVec.add_i(&dest[id_*nr_col],
        	&src[i*nr_col], 1., nr_col)
 

@cython.cdivision(True)
cdef void _adam_momentum(weight_t* gradient, weight_t* mom1, weight_t* mom2,
        int nr_weight, weight_t beta1, weight_t beta2, weight_t eps,
        weight_t learn_rate) nogil:
    # Calculate Adam on CPU, fused.
    # Assumes the learning rate adustment is calculated by the caller;
    # a_t = learn_rate * sqrt(1-beta2**timestep) / (1-beta1**timestep)
    cdef weight_t one_minus_beta1 = 1-beta1
    cdef weight_t one_minus_beta2 = 1-beta2
    cdef weight_t m1, m2, g
    cdef int i
    # Blockwise implementation is a bit faster. Adam is slooow :(
    cdef weight_t[64] buff
    cdef int steps = nr_weight // 64
    if steps * 64 < nr_weight:
        steps += 1
    idx = 0
    for i in range(steps):
        step_size = min(64, nr_weight-idx)
        Vec.mul_i(mom1, beta1, step_size)
        VecVec.add_i(mom1, gradient, one_minus_beta1, step_size)
        Vec.mul_i(mom2, beta2, step_size)
        for j in range(step_size):
            mom2[j] += one_minus_beta2 * gradient[j] ** 2
        for j in range(step_size):
            buff[j] = sqrtf(mom2[j])
        for j in range(step_size):
            buff[j] += eps
        for j in range(step_size):
            buff[j] = mom1[j] / buff[j]
        for j in range(step_size):
            gradient[j] = buff[j]
        mom1 += step_size
        mom2 += step_size
        gradient += step_size
        idx += step_size


@cython.cdivision(True)
cdef void cpu_update_averages(weight_t* ema,
        const weight_t* weights, int nr_weight, weight_t t, weight_t max_decay) nogil:
    cdef weight_t decay = (1.0 + t) / (10.0 + t)
    if decay > max_decay:
        decay = max_decay
    cdef weight_t one_minus_decay = 1-decay
    cdef int i
    for i in range(nr_weight): # num_threads=4, schedule='static'):
        ema[i] -= one_minus_decay * (ema[i] - weights[i])


class CupyOps(Ops):
    device = 'gpu'
    xp = cupy

    def gemm(self, x, y, out=None, trans1=False, trans2=False):
        if trans1:
            x = x.T
        if trans2:
            y = y.T
        if out is None:
            return self.xp.dot(x, y)
        else:
            self.xp.dot(x, y, out=out)
            return out

    def asarray(self, X, dtype=None):
        if isinstance(X, cupy.ndarray):
            return self.xp.asarray(X, dtype=dtype)
        elif hasattr(X, 'data_ptr'):
            # Handles PyTorch Tensors
            pointer = cupy.cuda.MemoryPointer(X.data_ptr())
            shape = X.stride()
            array = self.xp.ndarray(shape, memptr=pointer, dtype=dtype)
            return array
        else:
            return self.xp.array(X, dtype=dtype)

    def maxout(self, X):
        amax = X.max(axis=-1)
        argmax = self.asarray(X.argmax(axis=-1), dtype='i')
        return amax, argmax

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

    def selu(self, X, inplace=True):
        cdef float scale = 1.0507009873554805
        cdef float alpha = 1.6732632423543772
        out = scale * self.xp.where(X>=0., X, alpha * (self.xp.exp(X)-1.))
        if inplace:
            copy_array(X, out)
        return out

    def backprop_selu(self, delta, signal_in,
            inplace=True):
        # Backprop the SELU transformation
        cdef float scale = 1.0507009873554805
        cdef float alpha = 1.6732632423543772
        out = delta * self.xp.where(signal_in >= 0, scale,
                scale * alpha * self.xp.exp(signal_in))
        if inplace:
            copy_array(delta, out)
        return out

    def clip_gradient(self, gradient, threshold):
        xp = get_array_module(gradient)
        grad_norm = xp.linalg.norm(gradient)
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
        cols[1:, 0] = seq[:-1]
        cols[:, 1] = seq
        cols[:-1, 2] = seq[1:]
        return cols.reshape((B, I * (2*nW+1)))

    def backprop_seq2col(self, dY, int nW):
        cdef int nF = nW*2+1
        cdef int B = dY.shape[0]
        cdef int I = dY.shape[1] / nF
        assert nF == 3, "TODO: Support variable window size"
        # Having trouble getting the kernel to work...
        dX = self.allocate((B, I))
        dY = dY.reshape((B, nF, I))
        dX[:-1] += dY[1:, 0]
        dX += dY[:, nW]
        dX[1:] += dY[:-1, 2]
        return dX

    def mean_pool(self, X, lengths):
        return gpu_ops.mean_pool(self, X, lengths)

    def backprop_mean_pool(self, d_means, lengths):
        return gpu_ops.backprop_mean_pool(self, d_means, lengths)

    def max_pool(self, X, lengths):
        return gpu_ops.max_pool(self, X, lengths)

    def backprop_max_pool(self, d_maxes, which, lengths):
        return gpu_ops.backprop_max_pool(self, d_maxes, which, lengths)

    def sum_pool(self, X, lengths):
        return gpu_ops.sum_pool(self, X, lengths)

    def backprop_sum_pool(self, d_sums, lengths):
        return gpu_ops.backprop_sum_pool(self, d_sums, lengths)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def hash(self, ids, uint64_t seed):
        return gpu_ops.hash(self, ids, seed)

    def scatter_add(self, out, ids, inputs):
        self.xp.scatter_add(out, ids, inputs)

    def adam(self, weights, gradient, mom1, mom2, beta1, beta2, eps,
                   learn_rate, mod_rate=1.):
        cupy.ElementwiseKernel(
            'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps',
            'T param, T m, T v',
            '''m += one_minus_beta1 * (grad - m);
               v += one_minus_beta2 * (grad * grad - v);
               param -= lr * m / (sqrt(v) + eps);''',
            'adam')(gradient, learn_rate, 1 - beta1, 1 - beta2,
                    eps, weights, mom1, mom2)
        gradient.fill(0)

    def normal_init(self, W, fan_in, inplace=True):
        scale = self.xp.sqrt(1. / fan_in)
        inits = self.xp.random.normal(scale=scale, size=int(prod(W.shape)))
        inits = inits.reshape(W.shape)
        if inplace:
            copy_array(W, inits)
            return W
        else:
            return inits


cdef void seq2col(float* output, const float* X, int B, int I, int nW) nogil:
    '''
    Let's say nW is 1 (it usually is). Then we want to take:

    1a 1b 1c
    2a 2b 2c
    3a 3b 3c

    And make

    __ __ __ 1a 1b 1c 2a 2b 2c
    1a 1b 1c 2a 2b 2c 3a 3b 3c
    2a 2b 2c 3a 3b 3c __ __ __

    Where __ is padding.
    '''
    nF = nW * 2 + 1
    cdef int oI = nW * I
    cdef int xI = 0
    cdef int stride = I*nW
    cdef int stride1 = I*(nW+1)
    for i in range(B-nW):
        memcpy(&output[oI],
            &X[xI], stride1 * sizeof(output[0]))
        oI += stride1
        memcpy(&output[oI],
            &X[xI], stride * sizeof(output[0]))
        oI += stride
        xI += I
    memcpy(&output[oI],
        &X[xI], stride * sizeof(output[0]))


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
    for i in range(B*O):
        which__bo[i] = Vec.arg_max(&cands__bop[i*P], P)
        best__bo[i] = cands__bop[i*P + which__bo[i]]


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
    cdef weight_t variance = noise_level / ((1 + timestep) ** 0.55)
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


cdef void cpu_sum_pool(float* sums__bo,
        const float* X__to, const int* lengths__b,
        int B, int T, int O) nogil:
    '''Compute sums of a batch of concatenated sequences, using the lengths.'''
    for length in lengths__b[:B]:
        for _ in range(length):
            VecVec.add_i(sums__bo,
                X__to, 1.0, O)
            X__to += O
        sums__bo += O


cdef void cpu_backprop_sum_pool(float* dX__to,
        const float* d_sums__bo, const int* lengths__b,
        int B, int T, int O) nogil:
    for length in lengths__b[:B]:
        for _ in range(length):
            VecVec.add_i(dX__to,
                d_sums__bo, 1.0, O)
            dX__to += O
        d_sums__bo += O


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


cdef inline float sigmoid(float X) nogil:
    return 1./(1. + expf(-X))


cdef inline float dsigmoid(float y) nogil:
    return y*(1-y)


cdef inline float dtanh(float y) nogil:
    return 1-y**2


cdef void cpu_lstm_gates_fwd(float* output, float* cells, float* gates,
        const float* prev, int B, int N) nogil:
    cdef float hf, hi, ho, hc
    cdef int i, b
    for b in range(B):
        for i in range(N):
            hf = sigmoid(gates[i*4+0])
            hi = sigmoid(gates[i*4+1])
            ho = sigmoid(gates[i*4+2])
            hc = tanhf(gates[i*4+3])
            cells[i] = hf * prev[i] + hi * hc
            output[i] = tanhf(cells[i]) * ho
            gates[i*4+0] = hf
            gates[i*4+1] = hi
            gates[i*4+2] = ho
            gates[i*4+3] = hc
        output += N
        gates += N*4
        prev += N
        cells += N


cdef void cpu_lstm_gates_bwd(float* d_cells, float* d_prev, float* d_gates,
        const float* d_output, const float* gates, const float* cells,
        const float* prev, int B, int N) nogil:
    cdef float hf, hi, ho, hc, c, ct, dh, dho, dc, dhf, dhi, dhc, dprev
    cdef int i, b
    for b in range(B):
        for i in range(N):
            hf = gates[i*4+0]
            hi = gates[i*4+1]
            ho = gates[i*4+2]
            hc = gates[i*4+3]
            c  = cells[i]
            ct = tanhf(cells[i])
            dh = d_output[i]
            # Gradient for ho and c in h = sigmoid(ho) * tanh(c)
            dho = ct     * dh * dsigmoid(ho)
            dc  = ho     * dh * dtanh(ct)
            dc += d_cells[i]  # Carry gradient from previous step

            # Gradient for hf, hi, hc, prev[i]
            # in c = sigmoid(hf) * prev[i] + sigmoid(hi) * tanh(hc)
            dhf   = dsigmoid(hf) * dc * prev[i]
            dhi   = dsigmoid(hi) * dc * hc
            dhc   = dtanh(hc)    * dc * hi
            dprev =                dc * hf

            d_gates[i*4+0] = dhf
            d_gates[i*4+1] = dhi
            d_gates[i*4+2] = dho
            d_gates[i*4+3] = dhc
            d_cells[i] = dc
            d_prev[i] = dprev
        d_cells += N
        d_prev += N
        d_output += N
        d_gates += N*4
        gates += N*4
        cells += N
        prev += N


