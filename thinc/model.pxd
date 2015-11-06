from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from .typedefs cimport feat_t, weight_t
from .features cimport Feature


cdef class LinearModel:
    cdef PreshMap weights
    cdef Pool mem

    cdef void set_scores(self, weight_t* scores, const Feature* feats, int nr_feat) nogil
