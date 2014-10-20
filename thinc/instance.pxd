from cymem.cymem cimport Pool

from .typedefs cimport *
from thinc.learner cimport LinearModel
from thinc.features cimport Extractor

cdef class Instance:
    cdef Pool mem
    cdef int n_context
    cdef int n_feats
    cdef int n_class
    cdef size_t* context
    cdef feat_t* feats
    cdef weight_t* values
    cdef weight_t* scores
    cdef class_t clas

    cpdef class_t classify(self, LinearModel model, size_t[:] context=*,
                           feat_t[:] feats=*, Extractor extractor=*)
