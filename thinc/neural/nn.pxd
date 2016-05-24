from ..typedefs cimport weight_t, feat_t, class_t
from ..structs cimport NeuralNetC, FeatureC, ExampleC
from ..extra.eg cimport Example

from cymem.cymem cimport Pool


cdef class NeuralNet:
    cdef readonly Pool mem
    cdef readonly Example eg
    cdef NeuralNetC c

    cdef void set_scoresC(self, weight_t* scores, const FeatureC* feats, int nr_feat) nogil

    cdef int updateC(self, const ExampleC* eg) except -1

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1

    cdef void update_batchC(self, ExampleC** egs, int nr_eg) except *
