# cython: cdivision=True
# cython: infer_types=True
# cython: profile=True
from typing import Optional
from collections.abc import Sized
import numpy

cimport cython
from libc.string cimport memcpy, memset
from libc.stdlib cimport calloc, malloc, free
from libc.stdint cimport uint32_t, uint64_t
from libc.string cimport memcpy
from libc.math cimport isnan
from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap
from murmurhash.mrmr cimport hash64
cimport numpy as np
cimport blis.cy

from .. import registry
from ..util import copy_array, get_array_module
from ..types import DeviceTypes, DTypes, Shape, ArrayXd
from .linalg cimport VecVec, Vec
from .ops import Ops

try:
    import blis.py
    has_blis = True
except ImportError:
    has_blis = False


ctypedef float weight_t


cdef extern from "math.h":
    float logf(float x) nogil
    float sqrtf(float x) nogil
    float expf(float x) nogil
    float tanhf(float x) nogil
    float sinf(float x) nogil
    float cosf(float x) nogil


@registry.ops("NumpyOps")
class NumpyOps(Ops):
    name = "numpy"
    xp = numpy

    def __init__(
        self,
        device_type: DeviceTypes = "cpu",
        device_id: int = -1,
        *,
        use_blis: bool = True
    ) -> None:
        self.device_type = device_type
        self.device_id = device_id
        self.use_blis = use_blis
        if self.use_blis and not has_blis:
            raise ValueError("BLIS support requires blis: pip install blis")

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

    def alloc(self, shape: Shape, *, dtype: Optional[DTypes] = "float32") -> ArrayXd:
        return self.xp.zeros(shape, dtype=dtype)

    def gemm(self, np.ndarray x, np.ndarray y, *, np.ndarray out=None, trans1=False, trans2=False):
        if x.ndim != 2:
            raise ValueError(f"Provided 'x' array should be 2-dimensional, but found {x.ndim} dimension(s).")
        if y.ndim != 2:
            raise ValueError(f"Provided 'y' array should be 2-dimensional, but found {y.ndim} dimension(s).")
        if not self.use_blis:  # delegate to base Ops
            return super().gemm(x, y, out=out, trans1=trans1, trans2=trans2)
        x = self.as_contig(x)
        y = self.as_contig(y)
        if out is not None:
            out = self.as_contig(out)
        return blis.py.gemm(x, y, out=out, trans1=trans1, trans2=trans2, beta=0.)

    def relu(self, np.ndarray X, inplace=False):
        cdef np.ndarray out = X if inplace else X.copy()
        cdef weight_t* data = <weight_t*>out.data
        cdef size_t size = out.size
        for i in range(size):
            if data[i] < 0:
                data[i] = 0.
        return out

    def backprop_relu(self, np.ndarray dY, np.ndarray Y, inplace=False):
        cdef np.ndarray dX = dY if inplace else dY.copy()
        cdef size_t size = dX.size
        cdef weight_t* dX_ptr = <weight_t*>dX.data
        cdef const weight_t* Y_ptr = <const weight_t*>Y.data
        for i in range(size):
            if Y_ptr[i] <= 0:
                dX_ptr[i] = 0.
        return dX

    def lstm_forward_training(
        self,
        np.ndarray params,
        np.ndarray H0,
        np.ndarray C0,
        np.ndarray X,
        np.ndarray size_at_t
    ):
        assert H0.shape[0] == C0.shape[0]
        assert H0.shape[1] == C0.shape[1]
        Y, fwd_state = lstm_forward_training(params, H0, C0, X, size_at_t)
        return Y, fwd_state

    def lstm_forward_inference(
        self,
        np.ndarray params,
        np.ndarray H0,
        np.ndarray C0,
        np.ndarray X,
        np.ndarray size_at_t
    ):
        Y, _ = lstm_forward_training(params, H0, C0, X, size_at_t)
        return Y

    def backprop_lstm(
            self, np.ndarray dY, np.ndarray lengths, np.ndarray params, fwd_state
    ):
        dX, d_params = backprop_lstm(dY, lengths, params, fwd_state)
        return dX, d_params

    def maxout(self, const float[:, :, ::1] X):
        cdef Pool mem = Pool()
        cdef int B = X.shape[0]
        cdef int O = X.shape[1]
        cdef int P = X.shape[2]

        cdef np.ndarray best = numpy.zeros((B, O), dtype='float32', order='C')
        cdef np.ndarray which = numpy.zeros((B, O), dtype='int32', order='C')
        if len(X) > 0:
            cpu_maxout(<float*>best.data, <int*>which.data,
                &X[0, 0, 0], B, O, P)
        return best, which

    def backprop_maxout(self, const float[:, ::1] dY, int[:, ::1] which, int P):
        cdef int B = dY.shape[0]
        cdef int O = dY.shape[1]

        cdef np.ndarray dX = numpy.zeros((B, O, P), dtype='float32')
        cpu_backprop_maxout(<float*>dX.data,
            &dY[0, 0], &which[0, 0], B, O, P)
        return dX

    def mish(self, np.ndarray X, threshold=20.0, inplace: bool = False):
        cdef np.ndarray Y
        if X.dtype == "float32":
            if inplace:
                Y = X
            else:
                Y = self.xp.empty_like(X)
            cpu_mish(<float*>Y.data, <float *>X.data, threshold, X.size)
            return Y
        else:
            return super().mish(X, threshold, inplace)

    def backprop_mish(self, np.ndarray dY, np.ndarray X, threshold=20.0, inplace=False):
        cdef np.ndarray dX
        if dY.dtype == "float32" and X.dtype == "float32":
            if inplace:
                dX = dY
            else:
                dX = self.xp.empty_like(X)
            cpu_backprop_mish(<float*>dX.data, <float*>dY.data, <float*>X.data, threshold, X.size)
            return dX
        else:
            return super().backprop_mish(dY, X, threshold, inplace)

    def seq2col(self, const float[:, ::1] seq, int nW, *, const int[::1] lengths=None):
        """Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1))
        sequence. The new sequence is constructed by concatenating nW preceding
        and succeeding vectors onto each column in the sequence, to extract a
         window of features.
        """
        cdef int B = seq.shape[0]
        cdef int I = seq.shape[1]

        lengths = check_seq2col_lengths(self, lengths, B)
        cdef int nL = lengths.shape[0]

        cdef np.ndarray cols = self.alloc((B, (2*nW + 1) * I), dtype="float32")

        if seq.size != 0 and lengths.size != 0:
            seq2col(<float*>cols.data, &seq[0,0], &lengths[0], nW, B, I, nL)

        return cols

    def backprop_seq2col(self, const float[:, ::1] dY, int nW, *, const int[::1] lengths=None):
        cdef int B = dY.shape[0]
        cdef int nF = nW*2+1
        cdef int I = dY.shape[1] / nF

        lengths = check_seq2col_lengths(self, lengths, B)
        cdef int nL = lengths.shape[0]

        cdef np.ndarray dX = self.alloc((B, I), dtype='float32')
        if dY.size != 0 and lengths.size != 0:
            backprop_seq2col(<float*>dX.data, &dY[0,0], &lengths[0], B, I, nW, nL)
        return dX

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def hash(self, const uint64_t[::1] ids, uint32_t seed):
        """Hash a sequence of 64-bit keys into a table with 4 32-bit keys."""
        # Written to mirror the GPU implementation
        cdef np.ndarray[uint32_t, ndim=2] keys = self.alloc((ids.shape[0], 4), dtype='uint32')
        cdef int i
        cdef uint32_t* dest = <uint32_t*>keys.data
        for i in range(len(ids)):
            MurmurHash3_x86_128_uint64(ids[i], seed, &dest[i*4])
        return keys

    def reduce_mean(self, const float[:, ::1] X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        cdef Pool mem = Pool()
        assert B != 0
        assert O != 0
        means = <float*>mem.alloc(B * O, sizeof(float))

        cpu_reduce_mean(means,
            &X[0, 0], &lengths[0], B, T, O)
        return cpu_floats_ptr2array(means, (B, O))

    def reduce_sum(self, const float[:, ::1] X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        cdef Pool mem = Pool()
        assert B != 0
        assert O != 0
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
        assert T != 0
        assert O != 0
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
        assert T != 0
        assert O != 0
        dX = <float*>mem.alloc(T * O, sizeof(float))

        cpu_backprop_reduce_sum(dX,
            &d_sums[0,0], &lengths[0], B, T, O)
        return cpu_floats_ptr2array(dX, (T, O))

    def reduce_max(self, const float[:, ::1] X, const int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        cdef Pool mem = Pool()
        assert B != 0
        assert O != 0
        maxes = <float*>mem.alloc(B * O, sizeof(float))
        which = <int*>mem.alloc(B * O, sizeof(int))

        cpu_reduce_max(maxes, which,
            &X[0, 0], &lengths[0], B, T, O)

        cdef np.ndarray py_best = cpu_floats_ptr2array(maxes, (B, O))
        cdef np.ndarray py_which = cpu_ints_ptr2array(which, (B, O))
        return py_best, py_which

    def backprop_reduce_max(self, const float[:, ::1] d_maxes,
            const int[:, ::1] which, const int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_maxes.shape[1]
        cdef int T = 0
        for length in lengths[:B]:
            T += length
        cdef Pool mem = Pool()
        assert T != 0
        assert O != 0
        dX = <float*>mem.alloc(T * O, sizeof(float))

        cpu_backprop_reduce_max(dX,
            &d_maxes[0,0], &which[0, 0], &lengths[0], B, T, O)

        return cpu_floats_ptr2array(dX, (T, O))

    def scatter_add(self, np.ndarray table, np.ndarray indices, np.ndarray values):
        if table.dtype == 'float32' \
        and indices.dtype == 'int32' \
        and values.dtype == 'float32' \
        and table.flags.c_contiguous \
        and indices.flags.c_contiguous \
        and values.flags.c_contiguous \
        and indices.ndim == 1 \
        and table.ndim == 2 \
        and values.ndim == 2 \
        and values.shape[0] == indices.shape[0] \
        and values.shape[1] == table.shape[1]:
            cpu_scatter_add(<float*>table.data,
                <int*>indices.data, <float*>values.data,
                indices.shape[0], table.shape[1])
        else:
            self.xp.add.at(table, indices, values)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def adam(self, np.ndarray weights, np.ndarray gradient, np.ndarray mom1,
             np.ndarray mom2, const float beta1, const float beta2, float eps,
            float learn_rate, float mod_rate=1.):
        _adam_momentum(<float*>gradient.data, <float*>mom1.data, <float*>mom2.data,
            weights.shape[0], beta1, beta2, eps, learn_rate)
        VecVec.add_i(<float*>weights.data,
            <float*>gradient.data, -learn_rate, weights.shape[0])
        memset(<float*>gradient.data, 0, gradient.size * sizeof(float))
        return weights, gradient, mom1, mom2

    def ngrams(self, int n, const uint64_t[::1] keys):
        if n < 1:
            return self.alloc((0,), dtype="uint64")
        keys_ = <uint64_t*>&keys[0]
        length = max(0, keys.shape[0]-(n-1))
        cdef np.ndarray output_ = self.alloc((length,), dtype="uint64")
        output = <uint64_t*>output_.data
        for i in range(keys.shape[0]-(n-1)):
            output[i] = hash64(&keys_[i], n*sizeof(keys_[0]), 0)
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


def check_seq2col_lengths(ops, lengths, B):
    if lengths is None:
        lengths = ops.asarray1i([B])
    else:
        assert ops.xp.all(ops.xp.array(lengths) >= 0), "All sequence lengths must be >= 0"
        assert ops.xp.sum(lengths) == B, "The lengths must sum up to the batch length"

    return lengths


cdef void seq2col(float* output, const float* X, const int* L, int nW, int B, int I, int nL) nogil:
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

    If lengths > 1, then the sequence lengths dictate
    the boundaries/padding rather than the begin/end
    of X.
    '''

    nF = nW * 2 + 1

    seq_start = 0
    for i in range(nL):
        # Calculate the bounds of the next sequence.
        seq_end = seq_start + L[i]

        # Four-argument range loop only works with constant step.
        for j in range(seq_start, seq_end):
            # Find the unconstrained window around b, which
            # may be out of the sequence bounds.
            window_start = j - nW
            window_end = j + nW + 1

            # Find the sequence-constrained window around b.
            x_start = max(seq_start, window_start)
            x_end = min(seq_end, window_end)
            n_elems = x_end - x_start

            out_offset = x_start - window_start

            memcpy(output + (j * nF * I) + (out_offset * I),
                   X + (x_start * I),
                   n_elems * I * sizeof(output[0]))

        seq_start += L[i]


cdef void backprop_seq2col(float* d_seqs,
        const float* d_cols, const int* L, int B, int I, int nW, int nL) nogil:
    # Here's what we're doing, if we had 2d indexing.
    #for i in range(B):
    #    d_seq[i] += d_cols[i-2, 4]
    #    d_seq[i] += d_cols[i-1, 3]
    #    d_seq[i] += d_cols[i, 2]
    #    d_seq[i] += d_cols[i+1, 1]
    #    d_seq[i] += d_cols[i+2, 0]

    nF = nW * 2 + 1

    seq_start = 0
    for i in range(nL):
        # Calculate the bounds of the next sequence.
        seq_end = seq_start + L[i]

        for j in range(seq_start, seq_end):
            # Find the unconstrained window around b, which
            # may be out of the sequence bounds.
            window_begin = j - nW
            window_end = j + nW + 1

            # Find the sequence-constrained window around b.
            d_seqs_begin = max(seq_start, window_begin)
            d_seqs_end = min(seq_end, window_end)
            n_elems = d_seqs_end - d_seqs_begin

            # If the left window is cut short, we want to
            # start by the same amount in the output.
            out_offset = d_seqs_begin - window_begin

            VecVec.add_i(&d_seqs[d_seqs_begin * I],
                         &d_cols[(j * nF * I) + (out_offset * I)],
                         1., n_elems * I)

        seq_start += L[i]


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
    # Assumes the learning rate adjustment is calculated by the caller;
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
    cdef np.ndarray py_out = numpy.zeros(shape, dtype='float32')
    cdef int N = numpy.prod(shape)
    memcpy(py_out.data, ptr, N * sizeof(ptr[0]))
    return py_out


cdef cpu_ints_ptr2array(int* ptr, shape):
    cdef np.ndarray py_out = numpy.zeros(shape, dtype='int32')
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


def lstm_forward_training(
    np.ndarray params, np.ndarray c_init, np.ndarray h_init,
    np.ndarray X, np.ndarray lengths
):
    xp = numpy
    depth = c_init.shape[0]
    dirs = c_init.shape[1]
    nO = c_init.shape[2]
    N = X.shape[0]
    nI = X.shape[1]
    nT = lengths.shape[0]
    cdef int batch_size = lengths[0]
    # Preallocate these so we can pass them through for loop.
    cdef np.ndarray G = xp.zeros((depth, dirs, X.shape[0], nO * 4), dtype="f")
    cdef np.ndarray Y = xp.zeros((depth, dirs, X.shape[0], nO), dtype="f")
    cdef np.ndarray C = xp.zeros((depth, dirs, X.shape[0], nO), dtype="f")
    cdef np.ndarray Yt2 = numpy.zeros((batch_size, nO), dtype="f")
    cdef np.ndarray Ct2 = numpy.zeros((batch_size, nO), dtype="f")

    cdef int params_i = 0
    cdef int seq_i = 0
    orig_X = X
    cdef int i
    cdef np.ndarray Yid
    cdef np.ndarray Cid
    cdef np.ndarray Gid
    cdef np.ndarray Wx
    cdef np.ndarray Wh
    cdef np.ndarray bias
    for i in range(depth):
        nI = X.shape[1]
        for d in range(dirs):
            # The inits are shaped (depth, dirs, nO). We add the internal dimension
            # to make them set correctly.
            Yt2[:] = h_init[i, d].reshape((1, nO))
            Ct2[:] = c_init[i, d].reshape((1, nO))
            layer_params, params_i = _split_weights(params, i, nO, nI, params_i)
            Wx, Wh, bias = _transpose_weights(layer_params)
            Yid = Y[i, d]
            Cid = C[i, d]
            Gid = G[i, d]
            _lstm_forward_training(
                d, N, nO, nI, nT, 
                Gid,
                <float*>Yid.data,
                <float*>Cid.data,
                <float*>X.data,
                <float*>Wx.data,
                <float*>Wh.data,
                bias,
                <int*>lengths.data,
                <float*>Yt2.data,
                <float*>Ct2.data
            )
        H = Y[i].transpose((1, 0, 2)).reshape((N, -1))
        if dirs == 2:
            H = xp.ascontiguousarray(H)
        X = H
    return H, (Y, G, C, orig_X)


cdef int _lstm_forward_training(
    int d, int N, int nO, int nI, int nT,
    np.ndarray G,
    float* Y,
    float* C,
    float* X,
    float* Wx,
    float* Wh,
    np.ndarray bias,
    int* lengths,
    float* Yt2,
    float* Ct2,
) except -1:
    cdef double one = 1.0
    blis.cy.gemm(blis.cy.NO_TRANSPOSE, blis.cy.TRANSPOSE,
        N, nO*4, nI,
        one,
        X, nI, 1,
        Wx, nI, 1,
        one,
        <float*>G.data, nO*4, 1
    )
    cdef int t, batch_size
    cdef int seq_i = 0 if d == 0 else N
    cdef int i, j
    cdef np.ndarray Gt3_
    for t in range(nT):
        if d == 0:
            batch_size = lengths[t]
        else:
            batch_size = lengths[nT-(t+1)]
            seq_i -= batch_size
        # Prepare the inputs
        Yt3 = &Y[seq_i*nO]
        Ct3 = &C[seq_i*nO]
        Gt3_ = G[seq_i : seq_i+batch_size]
        Gt3 = <float*>Gt3_.data
        # Now do the actual calculation
        blis.cy.gemm(blis.cy.NO_TRANSPOSE, blis.cy.TRANSPOSE,
            batch_size, nO*4, nO,
            one,
            Yt2, nO, 1,
            Wh, nO, 1,
            one,
            Gt3, nO*4, 1
        )
        # This is super weird: if we remove this add, it gets slower? I guess
        # it does cache prefetching or something?
        # It's annoying though --- it means I can't really refactor further,
        # because speed goes down if I remove this.
        Gt3_ += bias
        #for i in range(batch_size):
        #    for j in range(nO*4):
        #        Gt3[i*nO*4+j] += bias[j]
        cpu_lstm_activate_fwd(Gt3,
            batch_size, nO)
        cpu_lstm_gates_fwd(Yt3, Ct3,
            Gt3, Ct2, batch_size, nO)
        if d == 0:
            seq_i += batch_size
        # We need to keep a full-sized array here, padded with the sequence-start
        # values. This isn't necessary for the l2r part, but for the r2l part
        # it's necessary, as we otherwise would have the previous step smaller
        # than the current.
        memcpy(Yt2, Yt3, sizeof(Yt3[0]) * batch_size * nO)
        memcpy(Ct2, Ct3, sizeof(Ct3[0]) * batch_size * nO)


def backprop_lstm(np.ndarray dY, np.ndarray lengths, np.ndarray params, fwd_state):
    xp = numpy
    cdef np.ndarray Y
    cdef np.ndarray G
    cdef np.ndarray C
    cdef np.ndarray X
    cdef np.ndarray Yid
    cdef np.ndarray Cid
    cdef np.ndarray Gid
    cdef np.ndarray Wx, Wh, bias
    cdef np.ndarray dWx, dWh, d_bias
    cdef np.ndarray dYid
    Y, G, C, X = fwd_state
    cdef int depth = C.shape[0]
    cdef int dirs = C.shape[1]
    cdef int N = C.shape[2]
    cdef int nO = C.shape[3]
    cdef int nI = X.shape[1]
    cdef int batch_size = lengths[0]
    cdef int nT = lengths.shape[0]
    # We don't need to store all the cells for all the layers.
    cdef np.ndarray dC = xp.zeros((N, nO), dtype=C.dtype)
    cdef np.ndarray dG = xp.zeros((N, nO*4), dtype=C.dtype)
    cdef np.ndarray d_params = xp.zeros((params.shape[0],), dtype=params.dtype)
    # Collect the params and slices. It makes it a bit easier to get the indexing
    # right, when we're iterating backwards.
    params_i = 0
    all_layer_params = []
    for i in range(depth):
        all_layer_params.append([])
        n_inputs = nI if i == 0 else (nO * dirs)
        for d in range(dirs):
            layer_params, params_i = _split_weights(params, i, nO, n_inputs, params_i)
            layer_params = _transpose_weights(layer_params)
            all_layer_params[-1].append((layer_params, params_i))
    params_i = 0
    all_layer_grads = []
    for i in range(depth):
        all_layer_grads.append([])
        n_inputs = nI if i == 0 else (nO * dirs)
        for d in range(dirs):
            layer_grads, params_i = _split_weights(params, i, nO, n_inputs, params_i)
            layer_grads = _transpose_weights(layer_grads)
            all_layer_grads[-1].append((layer_grads, params_i))
    # Similarly, we want to compute the indices first
    indices = []
    seq_i = 0
    for batch_size in lengths:
        indices.append((seq_i, batch_size))
        seq_i += batch_size

    cdef np.ndarray dX
    Xs = [X] + [Y[i].transpose(1, 0, 2).reshape((N, -1)) for i in range(depth-1)]
    dXs = [xp.zeros((X.shape[0], X.shape[1]), dtype=X.dtype) for X in Xs]
    # Okay, now do the actual looping
    for i in reversed(range(depth)):
        dY = dY.reshape((N, dirs, nO)).transpose((1, 0, 2))
        dX = dXs[i]
        X = Xs[i]
        if dirs >= 2:
            dY = numpy.ascontiguousarray(dY)
        for d in range(dirs):
            Wx, Wh, bias = all_layer_params[i][d][0]
            dWx, dWh, d_bias = all_layer_grads[i][d][0]
            assert Wx.shape[1] == dWx.shape[1] == X.shape[1] == dX.shape[1], (Wx.shape[1], dWx.shape[1], X.shape[1], dX.shape[1])
            dYid = dY[d] 
            dC.fill(0.)
            dG.fill(0.)
            Cid = C[i, d]
            Gid = G[i, d]
            Yid = Y[i, d]
            assert (Cid.shape[0], Cid.shape[1]) == (N, nO)
            assert (Yid.shape[0], Yid.shape[1]) == (N, nO)
            assert (Gid.shape[0], Gid.shape[1]) == (N, nO*4)
            assert (dYid.shape[0], dYid.shape[1]) == (N, nO)
            assert (dC.shape[0], dC.shape[1]) == (N, nO)
            assert (dG.shape[0], dG.shape[1]) == (N, nO*4)
            _lstm_backward_training(d, N, nO, dX.shape[1], nT,
                <float*>dX.data,
                <float*>dYid.data,
                <float*>dC.data,
                <float*>dG.data,
                <float*>dWx.data,
                <float*>dWh.data,
                <float*>d_bias.data,
                <float*>Cid.data,
                <float*>Gid.data, 
                <float*>Yid.data,
                <float*>X.data,
                <float*>Wx.data,
                <float*>Wh.data,
                list(indices)
            )
        dY = dX
    assert dX.shape[1] == X.shape[1]
    grad_parts = []
    for layer_grads in all_layer_grads:
        for dir_grads, _ in layer_grads:
            grad_parts.append(_untranspose_unsplit_weights(dir_grads))
    return dX, numpy.concatenate(grad_parts)


def _split_directions(X, dirs):
    if dirs == 1:
        return [X]
    else:
        X_ = X.reshape((X.shape[0], -1, dirs))
        Xs = []
        for d in range(dirs):
            Xs.append(numpy.ascontiguousarray(X_[:, d]))
        return Xs


cdef int _lstm_backward_training(
    int d, int N, int nO, int nI, int nT,
    float* dX,
    float* dY,
    float* dC,
    float* dG,
    float* dWx,
    float* dWh,
    float* d_bias,
    const float* C,
    const float* G,
    const float* Y,
    const float* X,
    const float* Wx,
    const float* Wh,
    indices,
) except -1:
    cdef int seq_t2
    cdef int seq_t3
    cdef double one = 1.0
    if d == 0:
        seq_t3, size_t3 = indices[-1]
        indices = indices[:-1]
        indices.reverse()
    else:
        seq_t3, size_t3 = indices[0]
        indices = indices[1:]
    cdef int batch_size
    for seq_t2, size_t2 in indices:
        dGt3 = &dG[seq_t3*nO*4]
        dXt3 = &dX[seq_t3*nI]
        dYt3 = &dY[seq_t3*nO]
        dCt3 = &dC[seq_t3*nO]
        dYt2 = &dY[seq_t2*nO]
        dCt2 = &dC[seq_t2*nO]
        Ct3 = &C[seq_t3*nO]
        Gt3 = &G[seq_t3*nO*4]
        Ct2 = &C[seq_t2*nO]
        
        batch_size = min(size_t2, size_t3)
        cpu_lstm_gates_bwd(dGt3, dCt2,
            dYt3, dCt3, Gt3, Ct3, Ct2, batch_size * nO
        )
        # Backprop hidden-to-hidden w.r.t. hidden.
        #     dYt2 += dGt3 @ Wh
        blis.cy.gemm(blis.cy.NO_TRANSPOSE, blis.cy.NO_TRANSPOSE,
            batch_size, nO, nO*4,
            one,
            <float*>dGt3, nO*4, 1,
            <float*>Wh, nO, 1,
            one,
            dYt2, nO, 1
        )
        seq_t3 = seq_t2
        size_t3 = size_t2

    # Backprop input-to-hidden w.r.t. weights.
    #     dWx += dG @ X
    blis.cy.gemm(blis.cy.TRANSPOSE, blis.cy.NO_TRANSPOSE,
        nO*4, nI, N,
        one,
        <float*>dG, nO*4, 1,
        <float*>X, nI, 1,
        one,
        dWx, nI, 1
    )
    # Backprop hidden-to-hidden w.r.t weights.
    #     dWh += dG @ Y
    blis.cy.gemm(blis.cy.TRANSPOSE, blis.cy.NO_TRANSPOSE,
        nO*4, nO, N,
        one,
        <float*>dG, nO*4, 1,
        <float*>Y, nO, 1,
        one,
        dWh, nO, 1
    )
    # Backprop bias
    for i in range(N):
        for j in range(nO*4):
            d_bias[j] += dG[i*nO*4+j]

    # Backprop input-to-hidden w.r.t. input
    blis.cy.gemm(blis.cy.NO_TRANSPOSE, blis.cy.NO_TRANSPOSE,
        N, nI, nO*4,
        one,
        <float*>dG, nO*4, 1,
        <float*>Wx, nI, 1,
        one,
        dX, nI, 1
    )


def _split_weights(np.ndarray params, int i, int nO, int nI, int params_i):
    Wx_size = 4 * nO * nI
    bx_size = 4 * nO
    Wh_size = 4 * nO * nO
    bh_size = 4 * nO
    Wx = params[params_i : params_i + Wx_size].reshape((4 * nO, nI))
    params_i += Wx_size
    bx = params[params_i : params_i + bx_size].reshape((4 * nO,))
    params_i += bx_size
    Wh = params[params_i : params_i + Wh_size].reshape((4 * nO, nO))
    params_i += Wh_size
    bh = params[params_i : params_i + bh_size].reshape((4 * nO,))
    params_i += bh_size
    return ((Wx, bx), (Wh, bh)), params_i


def _transpose_weights(params):
    # Transpose the parameters so that the gates are the last dimension. This
    # makes it easier to fuse.
    (Wx, bx), (Wh, bh) = params
    Wx = Wx.reshape((4, -1, Wx.shape[-1]))
    Wx = Wx.transpose((1, 0, 2)).reshape((-1, Wx.shape[-1]))
    bx = bx.reshape((4, -1)).transpose((1, 0)).reshape((-1,))
    Wh = Wh.reshape((4, -1, Wh.shape[-1]))
    Wh = Wh.transpose((1, 0, 2)).reshape((-1, Wh.shape[-1]))
    bh = bh.reshape((4, -1)).transpose((1, 0)).reshape((-1,))
    ascontig = numpy.ascontiguousarray
    Wx = ascontig(Wx)
    Wh = ascontig(Wh)
    bias = ascontig(bx) + bh
    return Wx, Wh, bias


def _untranspose_unsplit_weights(params):
    Wx, Wh, bias = params
    nO = Wh.shape[1]
    nI = Wx.shape[1]
    Wx = Wx.reshape((-1, 4, nI)).transpose((1, 0, 2)).reshape((-1, nI))
    Wh = Wh.reshape((-1, 4, nO)).transpose((1, 0, 2)).reshape((-1, nO))
    bias = bias.reshape((-1, 4)).transpose((1, 0)).reshape((-1,))
    zeros = numpy.zeros(bias.shape, dtype="f")
    return numpy.concatenate((Wx.ravel(), bias, Wh.ravel(), zeros))


cdef inline float sigmoid(float X) nogil:
    return 1./(1. + expf(-X))


cdef inline float dsigmoid(float y) nogil:
    return y*(1-y)


cdef inline float dtanh(float y) nogil:
    return 1-y**2


cdef void cpu_lstm_activate_fwd(float* gates, int B, int N) nogil:
    """Apply sigmoid activation in-place to columns 0, 1, 2 and tanh to column 3.
    The data is assumed to have the gates in the last dimension.
    """
    # This just does the following, but unrolled slightly to give 
    # a better chance at simd.
    #
    # gates[g+i+0] = sigmoid(gates[g+i+0])
    # gates[g+i+1] = sigmoid(gates[g+i+1])
    # gates[g+i+2] = sigmoid(gates[g+i+2])
    # gates[g+i+3] = tanh(gates[g+i+3])
    #
    # I would've hoped the compiler would find this itself? It seems to make
    # it like, 10% faster. It feels like a dumb thing to do but it's not much
    # code. The problem with this sort of thing is it needs to be rebenchmarked
    # later...It's fine to revert this at a later date to the simpler loop.
    # Shrug. The weird thing is, why should the batch entries be a good loop
    # stride here? Surely something to do with cache lines would make more sense?
    cdef int i, b, g
    g = 0
    for b in range(B):
        g = b * N * 4
        end = g + N*4
        while g < end:
            gates[g+0] = expf(-gates[g+0])
            gates[g+1] = expf(-gates[g+1])
            gates[g+2] = expf(-gates[g+2])
            g += 4
        g = b * N * 4
        while g < end:
            gates[g+0] += 1
            gates[g+1] += 1
            gates[g+2] += 1
            g += 4
        g = b * N * 4
        while g < end:
            gates[g+0] = 1.0 / gates[g+0]
            gates[g+1] = 1.0 / gates[g+1]
            gates[g+2] = 1.0 / gates[g+2]
            g += 4
        g = b * N * 4
        while g < end:
            gates[g+3] = tanhf(gates[g+3])
            g += 4

 
cdef void cpu_lstm_gates_fwd(float* hiddens, float* cells,
        const float* gates, const float* prevcells, int B, int N) nogil:
    cdef float hf, hi, ho, hc, ct2, ct3
    cdef int i, b, g, c, h
    g = 0
    c = 0
    h = 0
    while g < B*N*4:
        hf = gates[g+0]
        hi = gates[g+1]
        ho = gates[g+2]
        hc = gates[g+3]
        ct2 = prevcells[c]
        ct3 = hf * ct2 + hi * hc
        hiddens[h] = tanhf(ct3) * ho
        cells[c] = ct3
        g += 4
        c += 1
        h += 1


cdef void cpu_lstm_gates_bwd(
    float* dGt3,
    float* dCt2,
    const float* dYt3,
    const float* dCt3,
    const float* Gt3,
    const float* Ct3,
    const float* Ct2,
    int N
) nogil:
    cdef int i
    cdef float ct2, ct3, hf, hi, ho, hc, tanh_ct3
    cdef float d_ho, d_tanh_ct3, dct3, d_hi, d_hc, d_hf
    for i in range(N):
        ct2 = Ct2[i]
        ct3 = Ct3[i]
        dct3 = dCt3[i]
        dyt3 = dYt3[i]
        hf = Gt3[i*4+0]
        hi = Gt3[i*4+1]
        ho = Gt3[i*4+2]
        hc = Gt3[i*4+3]
        
        tanh_ct3 = tanhf(ct3)
        # 3b: Yt3 = tanhCt3 * ho
        d_ho = dyt3 * tanh_ct3
        d_tanh_ct3 = dyt3 * ho
        # 3a: tanhCt3 = tanh(Ct3)
        dct3 += d_tanh_ct3 * dtanh(tanh_ct3)
        # 2b: Ct3 += hi * hc
        d_hi = dct3 * hc
        d_hc = dct3 * hi
        # 2a: Ct3 = hf * Ct2
        d_hf = dct3 * ct2
        dCt2[i] = dct3 * hf
        dGt3[i*4+0] = d_hf * dsigmoid(hf)  # 1a
        dGt3[i*4+1] = d_hi * dsigmoid(hi)  # 1b
        dGt3[i*4+2] = d_ho * dsigmoid(ho)  # 1c
        dGt3[i*4+3] = d_hc * dtanh(hc)  # 1d


cdef void MurmurHash3_x86_128_uint64(
    const uint64_t val,
    const uint32_t seed,
    uint32_t *out
) nogil:
    cdef uint64_t h1, h2

    h1 = val
    h1 *= 0x87c37b91114253d5ull
    h1 = (h1 << 31) | (h1 >> 33)
    h1 *= 0x4cf5ad432745937full
    h1 ^= seed
    h1 ^= 8
    h2 = seed
    h2 ^= 8
    h1 += h2
    h2 += h1
    h1 ^= h1 >> 33
    h1 *= 0xff51afd7ed558ccdull
    h1 ^= h1 >> 33
    h1 *= 0xc4ceb9fe1a85ec53ull
    h1 ^= h1 >> 33
    h2 ^= h2 >> 33
    h2 *= 0xff51afd7ed558ccdull
    h2 ^= h2 >> 33
    h2 *= 0xc4ceb9fe1a85ec53ull
    h2 ^= h2 >> 33
    h1 += h2
    h2 += h1

    out[0] = h1 & 0xffffffffu
    out[1] = h1 >> 32
    out[2] = h2 & 0xffffffffu
    out[3] = h2 >> 32
