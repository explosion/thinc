from cymem.cymem cimport Pool

from .structs cimport TemplateC, FeatureC
from .typedefs cimport atom_t


cdef class ConjunctionExtracter:
    cdef Pool mem
    cdef TemplateC* templates
    cdef readonly int nr_templ
    cdef readonly int nr_atom

    cdef int set_features(self, FeatureC* feats, const atom_t* atoms) nogil
