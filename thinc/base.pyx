# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True

from .extra.eg cimport Example
from . cimport prng


cdef class Model:
    def __call__(self, Example eg):
        raise NotImplementedError

    def train_example(self, Example eg):
        raise NotImplementedError

    def predict_example(self, Example eg):
        raise NotImplementedError

    def dump(self, loc):
        raise NotImplementedError

    def load(self, loc):
        raise NotImplementedError

    def end_training(self):
        pass

    @property
    def nr_feat(self):
        raise NotImplementedError

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        pass
    cdef void set_scoresC(self, weight_t* scores,
            const FeatureC* feats, int nr_feat) nogil:
        pass

    cdef void set_featuresC(self, ExampleC* eg, const void* state) nogil: 
        pass

    cdef void dropoutC(self, FeatureC* feats, weight_t drop_prob,
            int nr_feat) nogil:
        for i in range(nr_feat):
            if prng.uniform_double_PRN() < drop_prob:
                # Preserve the mean activation, by increasing the activation
                # of the non-dropped units. This way, we don't have to
                # re-balance the weights.
                # I think I read this somewhere.
                # If not...well, it makes sense right?
                # Right?
                feats[i].value *= 1.0 / drop_prob
            else:
                feats[i].value = 0
