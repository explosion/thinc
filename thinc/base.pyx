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
            const void* feats, int nr_feat, int is_sparse) nogil:
        pass

    cdef void set_featuresC(self, ExampleC* eg, const void* state) nogil: 
        pass

    cdef void dropoutC(self, void* _feats, weight_t keep_prob,
            int nr_feat, int is_sparse) nogil:
        if keep_prob == 1.0:
            return
        if is_sparse:
            sparse = <FeatureC*>_feats
        else:
            dense = <weight_t*>_feats
        for i in range(nr_feat):
            if prng.get_uniform() < keep_prob:
                # Preserve the mean activation, by increasing the activation
                # of the non-dropped units. This way, we don't have to
                # re-balance the weights.
                # I think I read this somewhere.
                # If not...well, it makes sense right?
                # Right?
                if is_sparse:
                    sparse[i].value *= 1.0 / keep_prob
                else:
                    dense[i] *= 1.0 / keep_prob
            else:
                if is_sparse:
                    sparse[i].value = 0
                else:
                    dense[i] = 0
