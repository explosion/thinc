from cymem.cymem cimport Pool

from .typedefs cimport weight_t, atom_t
from .features cimport Feature


cdef class Example:
    cdef Pool mem

    cdef int n_classes
    cdef int n_atoms
    cdef int n_features


    cdef atom_t* atoms
    cdef bint* is_valid
    cdef int* costs
    cdef weight_t* scores

    cdef int guess
    cdef int best
    cdef int cost
    cdef weight_t loss
