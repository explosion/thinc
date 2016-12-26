from cymem.cymem cimport Pool

from .typedefs cimport weight_t, feat_t, class_t
from .structs cimport ExampleC, FeatureC


cdef class Model:
    cdef void set_scoresC(self, weight_t* scores,
        const FeatureC* feats, int nr_feat) nogil
 
    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1

    cdef int set_featuresC(self, FeatureC* feats, const void* state) nogil 

    cdef void dropoutC(self, FeatureC* feats, weight_t drop_prob,
            int nr_feat) nogil
