from libc.stdint cimport int16_t, int32_t, uint64_t
from .typedefs cimport weight_t, atom_t


include "compile_time_constants.pxi"


cdef struct MatrixC:
    float[300 * 300] data
    int32_t nr_row
    int32_t nr_col


cdef struct LayerC:
    MatrixC W
    MatrixC b
    int32_t nr_in
    int32_t nr_out


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
