from cymem.cymem cimport Pool

from .typedefs cimport *

DEF MAX_TEMPLATE_LEN = 10


cdef struct Template:
    int[MAX_TEMPLATE_LEN] indices
    int length
    atom_t[MAX_TEMPLATE_LEN] atoms


cdef struct Feature:
    int i
    feat_t key
    weight_t value


cdef class Extractor:
    cdef Pool mem
    cdef Template* templates
    cdef Feature* feats
    cdef readonly int n_templ
    cdef Feature* get_feats(self, atom_t* atoms, int* length) nogil
    cdef int set_feats(self, Feature* feats, atom_t* atoms) nogil


cdef int count_feats(dict counts, Feature* feats, int n_feats, weight_t inc) except -1
