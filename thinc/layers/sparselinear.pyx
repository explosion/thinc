# cython: infer_types=True, cdivision=True, bounds_check=False, wraparound=False
from murmurhash.mrmr cimport hash32
cimport numpy as np
from libc.stdint cimport uint64_t, int32_t, uint32_t
cimport cython

from typing import Tuple, Callable, Optional

from ..types import ArrayXd
from ..model import Model
from ..config import registry
from ..util import get_width, is_cupy_array, is_numpy_array, get_array_module
from ..backends import NumpyOps, CupyOps


InT = Tuple[ArrayXd, ArrayXd, ArrayXd]
OutT = ArrayXd


@cython.binding(True)
@registry.layers("SparseLinear.v1")
def SparseLinear(nO: Optional[int] = None, length: int = 2 ** 18):
    # NB: We can't have generic return type annotation if we want function to
    # be bound (and inspectable): https://github.com/cython/cython/issues/2753
    return Model(
        "sparse_linear",
        forward,
        init=init,
        params={"W": None, "b": None},
        dims={"nO": nO, "length": length},
    )


@cython.binding(True)
def forward(model: Model, keys_values_lengths: InT, is_train: bool) -> Tuple[OutT, Callable]:
    # NB: We can't have generic Model annotation if we want function to
    # be bound (and inspectable): https://github.com/cython/cython/issues/2753
    keys, values, lengths = keys_values_lengths
    if is_cupy_array(keys):
        # Currently we don't have a GPU-compatible implementation of this function :(
        # It sucks, but at least we can get the correct result by copying to CPU.
        return _begin_gpu_update(model, keys, values, lengths)
    else:
        return _begin_cpu_update(model, keys, values, lengths)


def init(model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None) -> Model[InT, OutT]:
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    nO = model.get_dim("nO")
    length = model.get_dim("length")
    model.set_param("W", model.ops.alloc((nO * length,), dtype="f"))
    model.set_param("b", model.ops.alloc((nO,), dtype="f"))
    return model


def _begin_gpu_update(model: Model[InT, OutT], keys: ArrayXd, values: ArrayXd, lengths: ArrayXd) -> Tuple[ArrayXd, Callable]:
    xp = get_array_module(keys)
    scores_cpu, callback = _begin_cpu_update(model, keys.get(), values.get(), lengths.get())

    def backprop_gpu_update(d_scores: ArrayXd) -> Tuple[ArrayXd, ArrayXd, ArrayXd]:
        callback(d_scores.get())
        return (keys, values, lengths)

    return xp.asarray(scores_cpu), backprop_gpu_update


def _begin_cpu_update(model, np.ndarray keys, np.ndarray values, np.ndarray lengths):
    cdef int nO = model.get_dim("nO")
    cdef int length = model.get_dim("length")
    cdef np.ndarray W = model.get_param("W")
    cdef np.ndarray b = model.get_param("b")
    cdef np.ndarray scores = model.ops.alloc((len(lengths), nO))
    scores += b
    set_scoresC(<float*>scores.data,
        <uint64_t*>keys.data, <float*>values.data, <int32_t*>lengths.data,
        lengths.shape[0], nO,
        <float*>W.data, length)
    return scores, _finish_linear_update(model, keys, values, lengths)


class _finish_linear_update:
    """Move this out of a closure, into its own callable object, to avoid
    pickling errors :(."""
    def __init__(self, model, keys, values, lengths):
        self.model = model
        self.keys = keys
        self.values = values
        self.lengths = lengths

    def __call__(self, float[:, ::1] d_scores):
        nO = self.model.get_dim("nO")
        length = self.model.get_dim("length")
        cdef np.ndarray d_weights = self.model.ops.alloc((nO*length,))
        cdef np.ndarray d_bias = self.model.ops.alloc((nO,))
        cdef np.ndarray keys = self.keys
        cdef np.ndarray values = self.values
        cdef np.ndarray lengths = self.lengths
        set_gradientC(<float*>d_weights.data,
            <uint64_t*>keys.data, <float*>values.data, <int32_t*>lengths.data,
            lengths.shape[0], nO,
            &d_scores[0,0], length)
        cdef int i, j
        for i in range(d_scores.shape[0]):
            for j in range(d_scores.shape[1]):
                d_bias[j] += d_scores[i, j]
        self.model.inc_grad("W", d_weights)
        self.model.inc_grad("b", d_bias)
        return (self.keys, self.values, self.lengths)


cdef void set_scoresC(float* scores,
        const uint64_t* keys, const float* values, const int32_t* lengths,
        int batch_size, int nr_out,
        const float* weights, int nr_weight) nogil:
    cdef uint32_t idx1, idx2
    cdef uint32_t hash1, hash2
    for length in lengths[:batch_size]:
        for i in range(length):
            hash1 = hash32(<void*>&keys[i], sizeof(keys[i]), 0)
            hash2 = hash32(<void*>&keys[i], sizeof(keys[i]), 1)
            idx1 = hash1 & (nr_weight-1)
            idx2 = hash2 & (nr_weight-1)
            value = values[i]
            for clas in range(nr_out):
                scores[clas] += weights[idx1 + clas] * value
                scores[clas] += weights[idx2 + clas] * value
        scores += nr_out
        keys += length
        values += length


cdef void set_gradientC(float* d_weights,
        const uint64_t* keys, const float* values, const int32_t* lengths,
        int batch_size, int nr_out,
        const float* d_scores, int nr_weight) nogil:
    cdef uint32_t idx1, idx2
    cdef uint32_t hash1, hash2
    for length in lengths[:batch_size]:
        for i in range(length):
            hash1 = hash32(<void*>&keys[i], sizeof(keys[i]), 0)
            hash2 = hash32(<void*>&keys[i], sizeof(keys[i]), 1)
            idx1 = hash1 & (nr_weight-1)
            idx2 = hash2 & (nr_weight-1)
            value = values[i]
            for clas in range(nr_out):
                d_weights[idx1 + clas] += d_scores[clas] * value
                d_weights[idx2 + clas] += d_scores[clas] * value
        d_scores += nr_out
        keys += length
        values += length
