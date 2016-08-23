from cymem.cymem cimport Pool

from .typedefs cimport weight_t, feat_t, class_t
from .structs cimport ExampleC, FeatureC


cdef class Model:
    cdef void set_scoresC(self, weight_t* scores,
        const void* feats, int nr_feat, int is_sparse) nogil
 
    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1

    cdef void set_featuresC(self, ExampleC* eg, const void* state) nogil 

    cdef void dropoutC(self, void* feats, weight_t drop_prob,
            int nr_feat, int is_sparse) nogil
