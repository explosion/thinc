from cymem.cymem cimport Pool
cimport numpy as np

from .typedefs cimport weight_t, atom_t
from .features cimport Feature


cdef class Example:
    cdef Pool mem

    cdef int n_classes
    cdef int n_atoms
    cdef int n_features


    cdef np.ndarray is_valid
    cdef np.ndarray costs
    cdef np.ndarray scores
    cdef np.ndarray atoms
    cdef np.ndarray embeddings

    cdef int guess
    cdef int best
    cdef int cost
    cdef weight_t loss
