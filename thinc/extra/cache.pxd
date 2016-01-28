from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from ..typedefs cimport *

cdef class ScoresCache:
    cdef size_t i
    cdef size_t max_size
    cdef class_t scores_size
    cdef Pool mem
    cdef weight_t** _arrays
    cdef weight_t* _scores_if_full
    cdef PreshMap _cache
    cdef size_t n_hit
    cdef size_t n_total

    cdef weight_t* lookup(self, class_t size, void* kernel, bint* success)
