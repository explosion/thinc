from libc.stdint cimport int32_t
from .typedefs cimport weight_t


cdef struct SparseArrayC:
    int32_t key
    weight_t val

cdef class SparseArray:
    cdef SparseArrayC* c


cdef SparseArrayC* init(int key, weight_t value) except NULL


cdef int find_key(const SparseArrayC* array, int key) except -2


cdef SparseArrayC* resize(SparseArrayC* array) except NULL
