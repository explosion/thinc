from libc.stdint cimport int32_t, uint64_t
from .typedefs cimport weight_t, atom_t


DEF MAX_TEMPLATE_LEN = 10


cdef struct SparseArrayC:
    int32_t key
    weight_t val


cdef struct FeatureC:
    uint64_t key
    weight_t val


cdef struct SparseAverageC:
    SparseArrayC* curr
    SparseArrayC* avgs
    SparseArrayC* times


cdef struct TemplateC:
    int[MAX_TEMPLATE_LEN] indices
    int length
    atom_t[MAX_TEMPLATE_LEN] atoms
