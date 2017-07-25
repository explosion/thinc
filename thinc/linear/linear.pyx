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

@describe.attributes(
    nO=Dimension("Output size"),
    length=Dimension("Weights length"),
    W=Synapses("Weights matrix",
        lambda obj: (obj.nO * obj.length,),
        lambda W, ops: W.fill(0.)
    ),
    b=Biases("Biases",
        lambda obj: (obj.nO,),
    ),
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
        cdef uint64_t[::1] keys
        cdef long[::1] lengths
        cdef float[::1] values
        keys, values_, lengths = keys_values_lengths
        drop *= self.drop_factor
        mask = self.ops.get_dropout_mask(values_.shape, drop)
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
        def finish_update(float[:, ::1] d_scores, sgd=None):
            cdef float[::1] d_weights = self.d_W
            cdef float[::1] d_bias = self.d_b
            set_gradientC(&d_weights[0],
                &keys[0], &values[0], &lengths[0],
                lengths.shape[0], self.nO,
                &d_scores[0,0], self.length)
            cdef int i, j
            for i in range(d_scores.shape[0]):
                for j in range(d_scores.shape[1]):
                    d_bias[j] += d_scores[i, j]
            sgd(self._mem.weights, self._mem.gradient, key=self.id)
        return scores, finish_update


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
