from cymem.cymem cimport Pool

from .typedefs cimport weight_t, feat_t, class_t
from .structs cimport ExampleC


cdef class Model:
    cdef void set_scoresC(self, weight_t* scores,
        const void* feats, int nr_feat, int is_sparse) nogil
 
    cdef int updateC(self, const ExampleC* eg) except -1

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1
