from cymem.cymem cimport Pool

from .typedefs cimport *
from thinc.learner cimport LinearModel
from thinc.features cimport Extractor


cdef class Instance:
    cdef Pool mem
    cdef int n_context
    cdef int n_feats
    cdef int n_classes
    cdef size_t* atoms
    cdef feat_t* feats
    cdef weight_t* values
    cdef weight_t* scores
    cdef class_t clas
