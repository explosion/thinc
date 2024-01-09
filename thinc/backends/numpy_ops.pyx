# cython: cdivision=True
# cython: infer_types=True
from collections.abc import Sized
from typing import Optional

import numpy

cimport cython
cimport numpy as np
from cymem.cymem cimport Pool
from libc.math cimport isnan
from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport calloc, free, malloc
from libc.string cimport memcpy, memset
from murmurhash.mrmr cimport hash64
from preshed.maps cimport PreshMap

from .. import registry
from ..types import ArrayXd, DeviceTypes, DTypes, Shape
from ..util import copy_array, get_array_module

from .cblas cimport CBlas, daxpy, dgemm, saxpy, sgemm, sscal

from ..compat import has_blis
from .ops import Ops, _split_weights, _transpose_weights, _untranspose_unsplit_weights


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
            array = data
        elif hasattr(data, 'numpy'):
            # Handles PyTorch Tensor
            array = data.numpy()
        elif hasattr(data, "get"):
            array = data.get()
        else:
            array = self.xp.array(data, dtype=dtype)

        if dtype is not None:
            array = array.astype(dtype=dtype, copy=False)

        return array


    def alloc(self, shape: Shape, *, dtype: Optional[DTypes] = "float32", zeros: bool = True) -> ArrayXd:
        if zeros:
            return self.xp.zeros(shape, dtype=dtype)
        else:
            return self.xp.empty(shape, dtype=dtype)

    def cblas(self) -> CBlas:
        return CBlas()

    def gemm(self, np.ndarray x, np.ndarray y, *, np.ndarray out=None, trans1=False, trans2=False):
        if x.ndim != 2:
            raise ValueError(f"Provided 'x' array should be 2-dimensional, but found {x.ndim} dimension(s).")
        if y.ndim != 2:
            raise ValueError(f"Provided 'y' array should be 2-dimensional, but found {y.ndim} dimension(s).")
        if not self.use_blis:  # delegate to base Ops
            return super().gemm(x, y, out=out, trans1=trans1, trans2=trans2)

        x = self.as_contig(x)
        y = self.as_contig(y)

        cdef int nM = x.shape[0] if not trans1 else x.shape[1]
        cdef int nK = x.shape[1] if not trans1 else x.shape[0]
        cdef int nK_b = y.shape[0] if not trans2 else y.shape[1]
        cdef int nN = y.shape[1] if not trans2 else y.shape[0]
        if nK != nK_b:
            msg = "Shape mismatch for blis.gemm: (%d, %d), (%d, %d)"
            raise ValueError(msg % (nM, nK, nK_b, nN))

        if out is not None:
            out = self.as_contig(out)
        else:
            # Can be uninitialized as 'beta' is zero.
            out = numpy.empty((nM, nN), dtype=x.dtype)

        cblas = self.cblas()
        if x.dtype == "float32" and y.dtype == "float32" and out.dtype == "float32":
            sgemm(cblas)(trans1, trans2,
                nM, nN, nK,
                1.0,
                <float*>(x.data), x.shape[1],
                <float*>(y.data), y.shape[1],
                0.0,
                <float*>(out.data), out.shape[1])
        elif x.dtype == "float64" and y.dtype == "float64" and out.dtype == "float64":
            dgemm(cblas)(trans1, trans2,
                nM, nN, nK,
                1.0,
                <double*>(x.data), x.shape[1],
                <double*>(y.data), y.shape[1],
                0.0,
                <double*>(out.data), out.shape[1])
        else:
            raise ValueError(f"unsupported or mismatching array data types; got '{x.dtype}', '{y.dtype}', '{out.dtype}'")

        return out

    def relu(self, np.ndarray X, inplace=False):
        cdef np.ndarray Y

        if X.dtype == "float32":
            Y = _inplace_or_copy(X, inplace)
            cpu_relu(<float *>Y.data, <int>Y.size)
            return Y
        elif X.dtype == "float64":
            Y = _inplace_or_copy(X, inplace)
            cpu_relu(<double *>Y.data, <int>Y.size)
            return Y
        else:
            return super().relu(X, inplace=inplace)

    def backprop_relu(self, np.ndarray dY, np.ndarray Y, inplace=False):
        _check_compatible_shape(dY, Y)

        cdef size_t size = Y.size
        cdef float* dX_ptr
        cdef const float* Y_ptr = <const float*>Y.data
        cdef np.ndarray dX
        if dY.dtype == "float32" and Y.dtype == "float32":
            dX = _inplace_or_copy(dY, inplace)
            dX_ptr = <float*>dX.data
            for i in range(size):
                if Y_ptr[i] <= 0:
                    dX_ptr[i] = 0.
            return dX
        else:
            return super().backprop_relu(dY, Y, inplace)

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
        Y, fwd_state = lstm_forward_training(self.cblas(), params, H0, C0, X, size_at_t)
        return Y, fwd_state

    def lstm_forward_inference(
        self,
        np.ndarray params,
        np.ndarray H0,
        np.ndarray C0,
        np.ndarray X,
        np.ndarray size_at_t
    ):
        Y, _ = lstm_forward_training(self.cblas(), params, H0, C0, X, size_at_t)
        return Y

    def backprop_lstm(
            self, np.ndarray dY, np.ndarray lengths, np.ndarray params, fwd_state
    ):
        dX, d_params = backprop_lstm(self.cblas(), dY, lengths, params, fwd_state)
        return dX, d_params

    def maxout(self, reals3d_ft X):
        cdef int B = X.shape[0]
        cdef int O = X.shape[1]
        cdef int P = X.shape[2]

        cdef np.ndarray best
        cdef np.ndarray which = self.alloc(shape=(B, O), dtype='int32', zeros=False)
        if reals3d_ft is float3d_t:
            best = self.alloc(shape=(B, O), dtype="float32", zeros=False)
            if len(X) > 0:
                cpu_maxout(<float*>best.data, <int*>which.data,
                    &X[0, 0, 0], B, O, P)
        else:
            best = self.alloc(shape=(B, O), dtype="float64", zeros=False)
            if len(X) > 0:
                cpu_maxout(<double*>best.data, <int*>which.data,
                    &X[0, 0, 0], B, O, P)
        return best, which

    def backprop_maxout(self, reals2d_ft dY, int[:, ::1] which, int P):
        cdef int B = dY.shape[0]
        cdef int O = dY.shape[1]

        cdef np.ndarray dX
        if reals2d_ft == float2d_t:
            dX = numpy.zeros((B, O, P), dtype='float32')
            cpu_backprop_maxout(<float*>dX.data, <float*>&dY[0, 0], &which[0, 0], B, O, P)
        else:
            dX = numpy.zeros((B, O, P), dtype='float64')
            cpu_backprop_maxout(<double*>dX.data, <double*>&dY[0, 0], &which[0, 0], B, O, P)

        return dX

    def mish(self, np.ndarray X, threshold=20.0, inplace: bool = False):
        cdef np.ndarray Y

        if X.dtype == "float32":
            Y = _inplace_or_copy(X, inplace)
            cpu_mish(<float *>Y.data, <int>Y.size, <float>threshold)
            return Y
        elif X.dtype == "float64":
            Y = _inplace_or_copy(X, inplace)
            cpu_mish(<double *>Y.data, <int>Y.size, <double>threshold)
            return Y
        else:
            return super().mish(X, inplace=inplace)

    def backprop_mish(self, np.ndarray dY, np.ndarray X, threshold=20.0, inplace=False):
        _check_compatible_shape(dY, X)

        cdef np.ndarray dX

        if dY.dtype == "float32" and X.dtype == "float32":
            dX = _inplace_or_copy(dY, inplace)
            cpu_backprop_mish(<float*>dX.data, <float*>X.data, <int>X.size, <float>threshold)
            return dX
        elif dY.dtype == "float64" and X.dtype == "float64":
            dX = _inplace_or_copy(dY, inplace)
            cpu_backprop_mish(<double*>dX.data, <double*>X.data, <int>X.size, <double>threshold)
            return dX
        else:
            return super().backprop_mish(dY, X, threshold, inplace)

    def seq2col(self, np.ndarray seq, int nW, *, int[::1] lengths=None):
        """Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1))
        sequence. The new sequence is constructed by concatenating nW preceding
        and succeeding vectors onto each column in the sequence, to extract a
         window of features.
        """

        # Note: the type of seq should be changed to reals2d_ft once
        # cython/cython#4697 is fixed. The following checks can then be
        # removed, because they are guaranteed by the reals2d_ft
        # type.

        if seq.ndim != 2:
            msg = f"seq2col requires sequence array of dimensionality 2, was {seq.ndim}"
            raise ValueError(msg)
        if not seq.flags.c_contiguous:
            msg = f"seq2col requires sequence array that is in C order and contiguous"
            raise ValueError(msg)


        cdef int B = seq.shape[0]
        cdef int I = seq.shape[1]

        lengths = check_seq2col_lengths(self, lengths, B)
        cdef int nL = lengths.shape[0]

        cdef np.ndarray cols
        if seq.dtype == "float32":
            cols = self.alloc((B, (2*nW + 1) * I), dtype="float32")
            if seq.size != 0 and lengths.size != 0:
                seq2col(<float*>cols.data, <float*>seq.data, &lengths[0], nW, B, I, nL)
            return cols
        elif seq.dtype == "float64":
            cols = self.alloc((B, (2*nW + 1) * I), dtype="float64")
            if seq.size != 0 and lengths.size != 0:
                seq2col(<double*>cols.data, <double*>seq.data, &lengths[0], nW, B, I, nL)
            return cols
        else:
            return super().seq2col(seq, nW, lengths=lengths)

    def backprop_seq2col(self, np.ndarray dY, int nW, *, int[::1] lengths=None):
        # Note: the type of dY should be changed to reals2d_ft once
        # cython/cython#4697 is fixed. The following checks can then be
        # removed, because they are guaranteed by the reals2d_ft
        # type.

        if dY.ndim != 2:
            msg = f"backprop_seq2col requires gradient array of dimensionality 2, was {dY.ndim}"
            raise ValueError(msg)
        if not dY.flags.c_contiguous:
            msg = f"backprop_seq2col requires gradient array that is in C order and contiguous"
            raise ValueError(msg)

        cdef int B = dY.shape[0]
        cdef int nF = nW*2+1
        cdef int I = dY.shape[1] / nF

        lengths = check_seq2col_lengths(self, lengths, B)
        cdef int nL = lengths.shape[0]

        cdef np.ndarray dX
        if dY.dtype == "float32":
            dX = self.alloc((B, I), dtype='float32')
            if dY.size != 0 and lengths.size != 0:
                backprop_seq2col(<float*>dX.data, <float*>dY.data, &lengths[0], B, I, nW, nL)
            return dX
        elif dY.dtype == "float64":
            dX = self.alloc((B, I), dtype='float64')
            if dY.size != 0 and lengths.size != 0:
                backprop_seq2col(<double*>dX.data, <double*>dY.data, &lengths[0], B, I, nW, nL)
            return dX
        else:
            return super().backprop_seq2col(dY, nW, lengths=lengths)

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

    def reduce_mean(self, reals2d_ft X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        if B == 0 or O == 0:
            if reals2d_ft is float2d_t:
                return numpy.zeros(shape=(B, O), dtype="float32")
            else:
                return numpy.zeros(shape=(B, O), dtype="float64")

        cdef np.ndarray means
        if reals2d_ft is float2d_t:
            means = numpy.zeros(shape=(B, O), dtype="float32")
            cpu_reduce_mean(<float *>means.data, &X[0, 0], &lengths[0], B, T, O)
        else:
            means = numpy.zeros(shape=(B, O), dtype="float64")
            cpu_reduce_mean(<double *>means.data, &X[0, 0], &lengths[0], B, T, O)

        return means

    def backprop_reduce_mean(self, reals2d_ft d_means, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_means.shape[1]
        cdef int T = 0

        for length in lengths[:B]:
            if length < 0:
                raise ValueError(f"all sequence lengths must be >= 0, got {length}")
            T += length

        if T == 0 or O == 0:
            if reals2d_ft is float2d_t:
                return numpy.zeros(shape=(T, O), dtype="float32")
            else:
                return numpy.zeros(shape=(T, O), dtype="float64")

        cdef np.ndarray dX
        if reals2d_ft is float2d_t:
            dX = numpy.zeros((T, O), dtype="float32")
            cpu_backprop_reduce_mean(<float *>dX.data, &d_means[0,0], &lengths[0], B, T, O)
        else:
            dX = numpy.zeros((T, O), dtype="float64")
            cpu_backprop_reduce_mean(<double *>dX.data, &d_means[0,0], &lengths[0], B, T, O)

        return dX

    def reduce_sum(self, reals2d_ft X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        if B == 0 or O == 0:
            if reals2d_ft is float2d_t:
                return numpy.zeros(shape=(B, O), dtype="float32")
            else:
                return numpy.zeros(shape=(B, O), dtype="float64")

        cdef np.ndarray sums
        if reals2d_ft is float2d_t:
            sums = numpy.zeros(shape=(B, O), dtype="float32")
            cpu_reduce_sum(<float *>sums.data, &X[0, 0], &lengths[0], B, T, O)
        else:
            sums = numpy.zeros(shape=(B, O), dtype="float64")
            cpu_reduce_sum(<double *>sums.data, &X[0, 0], &lengths[0], B, T, O)

        return sums

    def backprop_reduce_sum(self, reals2d_ft d_sums, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_sums.shape[1]
        cdef int T = 0

        for length in lengths[:B]:
            if length < 0:
                raise ValueError(f"all sequence lengths must be >= 0, got {length}")
            T += length

        if T == 0 or O == 0:
            if reals2d_ft is float2d_t:
                return numpy.zeros(shape=(T, O), dtype="float32")
            else:
                return numpy.zeros(shape=(T, O), dtype="float64")

        cdef np.ndarray dX
        if reals2d_ft is float2d_t:
            dX = numpy.zeros((T, O), dtype="float32")
            cpu_backprop_reduce_sum(<float *>dX.data, &d_sums[0,0], &lengths[0], B, T, O)
        else:
            dX = numpy.zeros((T, O), dtype="float64")
            cpu_backprop_reduce_sum(<double *>dX.data, &d_sums[0,0], &lengths[0], B, T, O)

        return dX

    def reduce_max(self, reals2d_ft X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        # Needs to be zero-initialized as we start by assuming that the first element is the max value.
        cdef np.ndarray which = self.alloc(shape=(B, O), dtype="i", zeros=True)

        if B == 0 or O == 0:
            if reals2d_ft is float2d_t:
                return numpy.zeros(shape=(B, O), dtype="float32"), which
            else:
                return numpy.zeros(shape=(B, O), dtype="float64"), which

        cdef np.ndarray maxes
        if reals2d_ft is float2d_t:
            maxes = self.alloc(shape=(B, O), dtype="float32", zeros=False)
            cpu_reduce_max(<float*>maxes.data, <int*>which.data, &X[0, 0], &lengths[0], B, T, O)
        else:
            maxes = self.alloc(shape=(B, O), dtype="float64", zeros=False)
            cpu_reduce_max(<double*>maxes.data, <int*>which.data, &X[0, 0], &lengths[0], B, T, O)

        return maxes, which

    def backprop_reduce_max(self, reals2d_ft d_maxes, int[:, ::1] which, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_maxes.shape[1]
        cdef int T = 0

        for length in lengths[:B]:
            if length <= 0:
                raise ValueError(f"all sequence lengths must be > 0, got {length}")
            T += length

        if T == 0 or O == 0:
            if reals2d_ft is float2d_t:
                return numpy.zeros(shape=(T, O), dtype="float32")
            else:
                return numpy.zeros(shape=(T, O), dtype="float64")

        cdef np.ndarray dX
        if reals2d_ft is float2d_t:
            dX = numpy.zeros((T, O), dtype="float32")
            cpu_backprop_reduce_max(<float *>dX.data, &d_maxes[0,0], &which[0, 0],
                &lengths[0], B, T, O)
        else:
            dX = numpy.zeros((T, O), dtype="float64")
            cpu_backprop_reduce_max(<double *>dX.data, &d_maxes[0,0], &which[0, 0],
                &lengths[0], B, T, O)

        return dX

    def gather_add(self, reals2d_ft table, ints2d_ft indices):
        cdef CBlas cblas = self.cblas()
        rows = indices.shape[0]
        dims = table.shape[1]

        cdef np.ndarray output
        if reals2d_ft is float2d_t:
            output = self.xp.zeros((rows, dims), dtype="float32")
            cpu_gather_add(saxpy(cblas), <float *>output.data, &table[0, 0], &indices[0, 0],
                        table.shape[0], dims, rows, indices.shape[1])
        else:
            output = self.xp.zeros((rows, dims), dtype="float64")
            cpu_gather_add(daxpy(cblas), <double *>output.data, &table[0, 0], &indices[0, 0],
                        table.shape[0], dims, rows, indices.shape[1])

        return output

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
            cpu_scatter_add(self.cblas(), <float*>table.data,
                <int*>indices.data, <float*>values.data,
                indices.shape[0], table.shape[1])
        else:
            self.xp.add.at(table, indices, values)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def adam(self, np.ndarray[np.float32_t] weights, np.ndarray[np.float32_t] gradient,
            np.ndarray[np.float32_t] mom1, np.ndarray[np.float32_t] mom2,
            const float beta1, const float beta2, float eps,
            float learn_rate, float mod_rate=1.):
        _check_compatible_shape(weights, gradient)
        _check_compatible_shape(weights, mom1)
        _check_compatible_shape(weights, mom2)

        cdef CBlas cblas = self.cblas()
        _adam_momentum(cblas, <float*>gradient.data, <float*>mom1.data, <float*>mom2.data,
            weights.shape[0], beta1, beta2, eps, learn_rate)
        saxpy(cblas)(weights.shape[0], -learn_rate, <float*>gradient.data, 1, <float*>weights.data, 1)

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
            out_ = self.alloc((N, D), zeros=False)
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


cdef void cpu_scatter_add(CBlas cblas, float* dest,
        const int* indices, const float* src,
        int nr_id, int nr_col) nogil:
    cdef int i
    for i in range(nr_id):
        id_ = indices[i]
        if id_ >= 0:
            saxpy(cblas)(nr_col, 1., &src[i*nr_col], 1, &dest[id_*nr_col], 1)


@cython.cdivision(True)
cdef void _adam_momentum(CBlas cblas, float* gradient, float* mom1, float* mom2,
        int nr_weight, float beta1, float beta2, float eps,
        float learn_rate) nogil:
    # Calculate Adam on CPU, fused.
    # Assumes the learning rate adjustment is calculated by the caller;
    # a_t = learn_rate * sqrt(1-beta2**timestep) / (1-beta1**timestep)
    cdef float one_minus_beta1 = 1-beta1
    cdef float one_minus_beta2 = 1-beta2
    cdef float m1, m2, g
    cdef int i
    # Blockwise implementation is a bit faster. Adam is slooow :(
    cdef float[64] buff
    cdef int steps = nr_weight // 64
    if steps * 64 < nr_weight:
        steps += 1
    idx = 0
    for i in range(steps):
        step_size = min(64, nr_weight-idx)
        sscal(cblas)(step_size, beta1, mom1, 1)
        saxpy(cblas)(step_size, one_minus_beta1, gradient, 1, mom1, 1)
        sscal(cblas)(step_size, beta2, mom2, 1)
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


def lstm_forward_training(CBlas cblas,
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
                cblas,
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
    CBlas cblas,
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
    sgemm(cblas)(False, True,
        N, nO*4, nI,
        one,
        X, nI,
        Wx, nI,
        one,
        <float*>G.data, nO*4
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
        sgemm(cblas)(False, True,
            batch_size, nO*4, nO,
            one,
            Yt2, nO,
            Wh, nO,
            one,
            Gt3, nO*4
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


def backprop_lstm(CBlas cblas, np.ndarray dY, np.ndarray lengths, np.ndarray params, fwd_state):
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
            _lstm_backward_training(cblas, d, N, nO, dX.shape[1], nT,
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


cdef int _lstm_backward_training(
    CBlas cblas,
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
        sgemm(cblas)(False, False,
            batch_size, nO, nO*4,
            one,
            <float*>dGt3, nO*4,
            <float*>Wh, nO,
            one,
            dYt2, nO
        )
        seq_t3 = seq_t2
        size_t3 = size_t2

    # Backprop input-to-hidden w.r.t. weights.
    #     dWx += dG @ X
    sgemm(cblas)(True, False,
        nO*4, nI, N,
        one,
        <float*>dG, nO*4,
        <float*>X, nI,
        one,
        dWx, nI
    )
    # Backprop hidden-to-hidden w.r.t weights.
    #     dWh += dG @ Y
    sgemm(cblas)(True, False,
        nO*4, nO, N,
        one,
        <float*>dG, nO*4,
        <float*>Y, nO,
        one,
        dWh, nO
    )
    # Backprop bias
    for i in range(N):
        for j in range(nO*4):
            d_bias[j] += dG[i*nO*4+j]

    # Backprop input-to-hidden w.r.t. input
    sgemm(cblas)(False, False,
        N, nI, nO*4,
        one,
        <float*>dG, nO*4,
        <float*>Wx, nI,
        one,
        dX, nI
    )


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


def _check_compatible_shape(u: np.ndarray, v: np.ndarray):
    if u.shape != v.shape:
        msg = f"arrays have incompatible shapes: {u.shape} and {v.shape}"
        raise ValueError(msg)


cdef inline np.ndarray _inplace_or_copy(np.ndarray X, inplace):
    if inplace:
        return X
    else:
        return numpy.array(X)
