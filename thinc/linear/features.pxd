from cymem.cymem cimport Pool

from ..structs cimport TemplateC, FeatureC
from ..typedefs cimport atom_t


cdef class ConjunctionExtracter:
    cdef Pool mem
    cdef readonly int nr_templ
    cdef readonly int nr_embed
    cdef readonly int nr_atom
    cdef public int linear_mode

    cdef int set_features(self, FeatureC* feats, const atom_t* atoms) nogil
    
    cdef TemplateC* templates
    cdef object _py_templates
