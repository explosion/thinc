from cymem.cymem cimport Pool

from .typedefs cimport *
from thinc.ml.learner cimport LinearModel
from thinc.features.extractor cimport Extractor

cdef class Instance:
    cdef Pool mem
    cdef size_t* context
    cdef feat_t* feats
    cdef weight_t* values
    cdef weight_t* scores
    cdef class_t clas

    cpdef class_t classify(self, size_t[:] context, Extractor extractor, LinearModel model)


