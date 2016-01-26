from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from .typedefs cimport feat_t, weight_t
from .structs cimport FeatureC


cdef class Model:
    cdef PreshMap weights
    cdef Pool mem

    cdef void set_scores(self, weight_t* scores, const FeatureC* feats, int nr_feat) nogil


cdef class LinearModel(Model):
    pass
