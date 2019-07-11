# cython: infer_types=True
# cython: cdivision=True
# cython: bounds_check=False
# cython: wraparound=False
from murmurhash.mrmr cimport hash32
cimport numpy as np
from libc.stdint cimport uint64_t, int32_t, uint32_t
from ..neural._classes.model import Model
from .. import describe
from ..describe import Dimension, Synapses, Biases, Gradient
from ..neural.util import is_cupy_array, is_numpy_array

# Sigh, these stuff up pickling if they're lambdas...

def _get_W_shape(obj):
    return (obj.nO * obj.length,)

def _init_W(W, ops):
    W.fill(0.)

def _get_bias_shape(obj):
    return (obj.nO,)


@describe.attributes(
    nO=Dimension("Output size"),
    length=Dimension("Weights length"),
    W=Synapses("Weights matrix", _get_W_shape, _init_W),
    b=Biases("Biases", _get_bias_shape),
    d_W=Gradient("W"),
    d_b=Gradient("b"),
)
class LinearModel(Model):
    name = 'linear'
    def __init__(self, nO, length=2**18, **kwargs):
        Model.__init__(self, **kwargs)
        self.nO = nO
        self.length = length

    def begin_update(self, keys_values_lengths, drop=0.):
        keys, values, lengths = keys_values_lengths
        if is_cupy_array(keys):
            return self._begin_gpu_update(keys, values, lengths, drop=drop)
        else:
            return self._begin_cpu_update(keys, values, lengths, drop=drop)

    def _begin_gpu_update(self, keys, values, lengths, drop):
        # Currently we don't have a GPU-compatible implementation of this function :(
        # It sucks, but at least we can get the correct result by copying to CPU.
        cpu_keys = keys.get()
        cpu_values = values.get()
        cpu_lengths = lengths.get()
        return self._begin_cpu_update(cpu_keys, cpu_values, cpu_lengths, drop=drop)

    def _begin_cpu_update(self, uint64_t[::1] keys, np.ndarray values_, long[::1] lengths, drop):
        if drop is not None:
            drop *= self.drop_factor
        mask = self.ops.get_dropout_mask((values_.shape[0],), drop)
        cdef float[::1] values
        if mask is not None:
            values = values_ * mask
        else:
            values = values_
        cdef float[:, ::1] scores = self.ops.allocate((len(lengths), self.nO)) + self.b
        cdef float[::1] weights = self.W
        set_scoresC(&scores[0, 0],
            &keys[0], &values[0], &lengths[0],
            lengths.shape[0], self.nO,
            &weights[0], self.length)
        return scores, _finish_linear_update(self, keys, values, lengths)


class _finish_linear_update(object):
    """Move this out of a closure, into its own callable object, to avoid
    pickling errors :(."""
    def __init__(self, layer, keys, values, lengths):
        self.layer = layer
        self.keys = keys
        self.values = values
        self.lengths = lengths

    def __call__(self, float[:, ::1] d_scores, sgd=None):
        cdef float[::1] d_weights = self.layer.d_W
        cdef float[::1] d_bias = self.layer.d_b
        cdef uint64_t[::1] keys = self.keys
        cdef float[::1] values = self.values
        cdef long[::1] lengths = self.lengths
        set_gradientC(&d_weights[0],
            &keys[0], &values[0], &lengths[0],
            lengths.shape[0], self.layer.nO,
            &d_scores[0,0], self.layer.length)
        cdef int i, j
        for i in range(d_scores.shape[0]):
            for j in range(d_scores.shape[1]):
                d_bias[j] += d_scores[i, j]
        if sgd is not None:
            sgd(self.layer._mem.weights, self.layer._mem.gradient, key=self.layer.id)
        return None


cdef void set_scoresC(float* scores,
        const uint64_t* keys, const float* values, const long* lengths,
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
        const uint64_t* keys, const float* values, const long* lengths,
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
        #values += length
