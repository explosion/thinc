# cython: cdivision=True, infer_types=True, profile=True
cimport cython
from libc.string cimport memcpy, memset
from libc.stdlib cimport calloc, malloc, free
from libc.stdint cimport uint32_t, uint64_t
from libc.string cimport memcpy
from libc.math cimport isnan
from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap
import numpy
from numpy import prod
from numpy cimport ndarray
from collections.abc import Sized
cimport numpy as np
from murmurhash.mrmr cimport hash64, hash128_x86, hash128_x64

from ..util import copy_array, get_array_module
from .linalg cimport VecVec, Vec
from .ops import Ops

cimport blis
cimport blis.cy
import blis.py


ctypedef float weight_t


cdef extern from "math.h":
    float logf(float x) nogil
    float sqrtf(float x) nogil
    float expf(float x) nogil
    float tanhf(float x) nogil
    float sinf(float x) nogil
    float cosf(float x) nogil


class NumpyOps(Ops):
    device = 'cpu'
    xp = numpy

    def asarray(self, data, dtype=None):
        if isinstance(data, self.xp.ndarray):
            if dtype is not None:
                return self.xp.asarray(data, dtype=dtype)
            else:
                return self.xp.asarray(data)
        elif hasattr(data, 'numpy'):
            # Handles PyTorch Tensor
            return data.numpy()
        elif hasattr(data, "get"):
            return data.get()
        elif dtype is not None:
            return self.xp.array(data, dtype=dtype)
        else:
            return self.xp.array(data)

    def alloc(self, shape, dtype='float32'):
        return self.xp.zeros(shape, dtype=dtype)

    def gemm(self, const float[:, ::1] x, const float[:, ::1] y, trans1=False, trans2=False,
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
            out_array = self.alloc((m, n))
        else:
            out_array = self.xp.asarray(out)
        assert out_array.shape[0] == m
        assert out_array.shape[1] == n
        blis.py.gemm(x, y, out=out_array, trans1=trans1, trans2=trans2)
        return out_array

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

    def maxout(self, const float[:, :, ::1] py_cands):
        cdef Pool mem = Pool()
        cdef int B = py_cands.shape[0]
        cdef int O = py_cands.shape[1]
        cdef int P = py_cands.shape[2]

        cdef ndarray best = numpy.zeros((B, O), dtype='float32', order='C')
        cdef ndarray which = numpy.zeros((B, O), dtype='int32', order='C')
        cpu_maxout(<float*>best.data, <int*>which.data,
            &py_cands[0, 0, 0], B, O, P)
        return best, which

    def backprop_maxout(self, const float[:, ::1] dX__bo, int[:, ::1] which__bo, int P):
        cdef int B = dX__bo.shape[0]
        cdef int O = dX__bo.shape[1]

        cdef np.ndarray dX__bop = numpy.zeros((B, O, P), dtype='float32')
        cpu_backprop_maxout(<float*>dX__bop.data,
            &dX__bo[0, 0], &which__bo[0, 0], B, O, P)
        return dX__bop

    def mish(self, const float[:, ::1] X, threshold=5, out=None):
        shape = [X.shape[i] for i in range(X.ndim)]
        cdef np.ndarray Y = self.alloc(tuple(shape), dtype="f")
        cpu_mish(<float*>Y.data,
            &X[0, 0], threshold, X.size)
        if out is not None:
            out[:] = Y
            return out
        else:
            return Y

    def backprop_mish(self, const float[:, ::1] dY, const float[:, ::1] X,
            threshold=5, out=None):
        shape = [X.shape[i] for i in range(X.ndim)]
        cdef np.ndarray dX = self.alloc(tuple(shape), dtype="f")
        cpu_backprop_mish(<float*>dX.data,
            &dY[0, 0], &X[0, 0], threshold, X.size)
        if out is not None:
            out[:] = dX
            return out
        else:
            return dX

    def lstm(self, float[:, ::1] output, float[:, ::1] cells,
            float[:, :, ::1] gates, float[:, ::1] prev):
        cpu_lstm_gates_fwd(&output[0, 0], &cells[0, 0],
            &gates[0, 0, 0], &prev[0, 0], cells.shape[0], cells.shape[1])
        return output

    def backprop_lstm(self, float[:, ::1] d_cells, float[:, ::1] d_prev,
            float[:, :, ::1] d_gates, float[:, ::1] d_output,
            float[:, :, ::1] gates, float[:, ::1] cells, float[:, ::1] prev):
        cpu_lstm_gates_bwd(&d_cells[0, 0], &d_prev[0, 0], &d_gates[0, 0, 0],
            &d_output[0, 0], &gates[0, 0, 0], &cells[0, 0], &prev[0, 0],
            cells.shape[0], cells.shape[1])

    def seq2col(self, const float[:, ::1] seq, int nW):
        '''Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1)) sequence.
        The new sequence is constructed by concatenating nW preceding and succeeding
        vectors onto each column in the sequence, to extract a window of features.
        '''
        cdef int B = seq.shape[0]
        cdef int I = seq.shape[1]
        cdef ndarray cols = self.alloc((B, (2*nW + 1) * I), dtype='float32')
        seq2col(<float*>cols.data, &seq[0,0], nW, B, I)
        return cols

    def backprop_seq2col(self, const float[:, ::1] dY, int nW):
        cdef int B = dY.shape[0]
        cdef int nF = nW*2+1
        cdef int I = dY.shape[1] / nF
        cdef ndarray dX = self.alloc((B, I), dtype='float32')
        backprop_seq2col(<float*>dX.data, &dY[0,0], B, I, nW)
        return dX

    def remap_ids(self, PreshMap mapping, uint64_t[::1] ids_mv, uint64_t value=0):
        cdef uint64_t* ids = &ids_mv[0]
        cdef ndarray[uint64_t] output_arr = self.alloc(len(ids_mv), dtype='uint64')
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
    def hash(self, const uint64_t[::1] ids, uint32_t seed):
        '''Hash a sequence of 64-bit keys into a table with 4 32-bit keys'''
        # Written to mirror the GPU implementation
        cdef ndarray[uint32_t, ndim=2] keys = self.alloc((ids.shape[0], 4), dtype='uint32')
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

    def reduce_mean(self, const float[:, ::1] X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        cdef Pool mem = Pool()
        means = <float*>mem.alloc(B * O, sizeof(float))

        cpu_reduce_mean(means,
            &X[0, 0], &lengths[0], B, T, O)
        return cpu_floats_ptr2array(means, (B, O))

    def reduce_sum(self, const float[:, ::1] X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        cdef Pool mem = Pool()
        sums = <float*>mem.alloc(B * O, sizeof(float))

        cpu_reduce_sum(sums,
            &X[0, 0], &lengths[0], B, T, O)
        return cpu_floats_ptr2array(sums, (B, O))

    def backprop_reduce_mean(self, const float[:, ::1] d_means, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_means.shape[1]
        cdef int T = 0
        for length in lengths[:B]:
            T += length
        cdef Pool mem = Pool()
        dX = <float*>mem.alloc(T * O, sizeof(float))

        cpu_backprop_reduce_mean(dX,
            &d_means[0,0], &lengths[0], B, T, O)

        return cpu_floats_ptr2array(dX, (T, O))

    def backprop_reduce_sum(self, const float[:, ::1] d_sums, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_sums.shape[1]
        cdef int T = 0
        for length in lengths[:B]:
            T += length
        cdef Pool mem = Pool()
        dX = <float*>mem.alloc(T * O, sizeof(float))

        cpu_backprop_reduce_sum(dX,
            &d_sums[0,0], &lengths[0], B, T, O)
        return cpu_floats_ptr2array(dX, (T, O))

    def reduce_max(self, const float[:, ::1] X, const int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        cdef Pool mem = Pool()
        maxes = <float*>mem.alloc(B * O, sizeof(float))
        which = <int*>mem.alloc(B * O, sizeof(int))

        cpu_reduce_max(maxes, which,
            &X[0, 0], &lengths[0], B, T, O)

        cdef ndarray py_best = cpu_floats_ptr2array(maxes, (B, O))
        cdef ndarray py_which = cpu_ints_ptr2array(which, (B, O))
        return py_best, py_which

    def backprop_reduce_max(self, const float[:, ::1] d_maxes,
            const int[:, ::1] which, const int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_maxes.shape[1]
        cdef int T = 0
        for length in lengths[:B]:
            T += length
        cdef Pool mem = Pool()
        dX = <float*>mem.alloc(T * O, sizeof(float))

        cpu_backprop_reduce_max(dX,
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
             float[::1] mom2, const float beta1, const float beta2, float eps,
            float learn_rate, float mod_rate=1.):
        _adam_momentum(&gradient[0], &mom1[0], &mom2[0],
            weights.shape[0], beta1, beta2, eps, learn_rate)
        VecVec.add_i(&weights[0],
            &gradient[0], -learn_rate, weights.shape[0])
        memset(&gradient[0], 0, gradient.size * sizeof(float))

    def ngrams(self, int n, const uint64_t[::1] keys_):
        keys = <uint64_t*>&keys_[0]
        length = max(0, keys_.shape[0]-n)
        cdef np.ndarray output_ = self.alloc((length,), dtype='uint64')
        output = <uint64_t*>output_.data
        for i in range(keys_.shape[0]-n):
            output[i] = hash64(&keys[i], n*sizeof(keys[0]), 0)
        return output_

    def position_encode(self, int N, int D, int period=10000, out=None):
        cdef np.ndarray out_
        if out is None:
            out_ = self.alloc((N, D))
        else:
            out_ = out
        assert out_.shape[0] == N
        assert out_.shape[1] == D
        cpu_position_encode(<float*>out_.data, period, N, D)
        return out_


cdef void seq2col(float* output, const float* X, int nW, int B, int I) nogil:
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

    Now let's say nW is 2. Then we want to take:

    1a 1b 1c
    2a 2b 2c
    3a 3b 3c

    And make

    __ __ __ __ __ __ 1a 1b 1c 2a 2b 2c 3a 3b 3c
    __ __ __ 1a 1b 1c 2a 2b 2c 3a 3b 3c __ __ __
    1a 1b 1c 2a 2b 2c 3a 3b 3c __ __ __ __ __ __

    * x_start=-6, x_end=9 : (0-2) * 3, (0+2+1) * 3
    * x_start=-3, x_end=13 : (1-2) * 3, (1+2+1) * 3
    * x_start=0, x_end=16 : (2-2) * 3, (2+2+1) * 3

    '''
    nF = nW * 2 + 1
    for i in range(B):
        o_start = i * I * nF
        x_start = (i-nW) * I
        x_end = (i+nW+1) * I
        if x_start < 0:
            o_start += -x_start
            x_start = 0
        if x_end >= B * I:
            x_end = B * I
        memcpy(&output[o_start],
            &X[x_start], (x_end-x_start) * sizeof(output[0]))


cdef void backprop_seq2col(float* d_seqs,
        const float* d_cols, int B, int I, int nW) nogil:
    # Here's what we're doing, if we had 2d indexing.
    #for i in range(B):
    #    d_seq[i] += d_cols[i-2, 4]
    #    d_seq[i] += d_cols[i-1, 3]
    #    d_seq[i] += d_cols[i, 2]
    #    d_seq[i] += d_cols[i+1, 1]
    #    d_seq[i] += d_cols[i+2, 0]
    cdef int col_feat
    nF = nW * 2 + 1
    for i in range(B):
        seq_row = i * I
        col_feat = nF * I
        for f in range(-nW, nW+1):
            col_row = (i+f) * (I * nF)
            col_feat -= I
            if col_row >= 0 and (col_row < (B*I*nF)):
                j = col_row + col_feat
                if j >= 0 and (j+I) < (B*I*nF):
                    VecVec.add_i(&d_seqs[seq_row],
                        &d_cols[j], 1., I)


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


cdef void cpu_position_encode(float* output, float period, int N, int D) nogil:
    cdef float pos, d
    cdef int j
    cdef float dimensions = D
    for i in range(N):
        pos = i
        j = 0
        d = 0
        while (j+1) < D:
            d = j
            output[j]   = sinf(pos / period ** (2 * d / dimensions))
            output[j+1] = cosf(pos / period ** (2 * d / dimensions))
            j += 2
        if j < D:
            output[j]   = sinf(pos / period ** (2 * d / dimensions))
        output += D


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


cdef void cpu_mish(weight_t* Y, const weight_t* X, float threshold, int N) nogil:
    cdef float one = 1.
    for i in range(N):
        if X[i] >= threshold:
            Y[i] = X[i]
        else:
            Y[i] = X[i] * tanhf(logf(one + expf(X[i])))


cdef void cpu_backprop_mish(weight_t* dX,
        const weight_t* dY, const weight_t* X, float threshold, int N) nogil:
    cdef float one = 1.
    cdef float exp_x, exp_2x, exp_3x, omega, delta
    for i in range(N):
        x = X[i]
        if x >= threshold:
            dX[i] = dY[i]
        else:
            exp_x = expf(x)
            exp_2x = expf(2*x)
            exp_3x = expf(3*x)
            omega = (4. * (x+1)) + (4 * exp_2x) + exp_3x + exp_x * (4.*x+6)
            delta = 2. * exp_x + exp_2x + 2.
            dX[i] = dY[i] * ((exp_x * omega) / (delta * delta))


cdef cpu_floats_ptr2array(float* ptr, shape):
    cdef ndarray py_out = numpy.zeros(shape, dtype='float32')
    cdef int N = numpy.prod(shape)
    memcpy(py_out.data, ptr, N * sizeof(ptr[0]))
    return py_out


cdef cpu_ints_ptr2array(int* ptr, shape):
    cdef ndarray py_out = numpy.zeros(shape, dtype='int32')
    cdef int N = numpy.prod(shape)
    memcpy(py_out.data, ptr, N * sizeof(ptr[0]))
    return py_out


cdef void cpu_reduce_mean(float* means__bo,
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


cdef void cpu_backprop_reduce_mean(float* dX__to,
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


cdef void cpu_reduce_sum(float* sums__bo,
        const float* X__to, const int* lengths__b,
        int B, int T, int O) nogil:
    '''Compute sums of a batch of concatenated sequences, using the lengths.'''
    for length in lengths__b[:B]:
        for _ in range(length):
            VecVec.add_i(sums__bo,
                X__to, 1.0, O)
            X__to += O
        sums__bo += O


cdef void cpu_backprop_reduce_sum(float* dX__to,
        const float* d_sums__bo, const int* lengths__b,
        int B, int T, int O) nogil:
    for length in lengths__b[:B]:
        for _ in range(length):
            VecVec.add_i(dX__to,
                d_sums__bo, 1.0, O)
            dX__to += O
        d_sums__bo += O


cdef void cpu_reduce_max(float* maxes__bo, int* which__bo,
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


cdef void cpu_backprop_reduce_max(float* dX__to,
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
