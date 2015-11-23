from libc.stdint cimport int16_t, int32_t, uint64_t
from .typedefs cimport weight_t, atom_t


include "compile_time_constants.pxi"


cdef struct MatrixC:
    float* data
    int32_t nr_row
    int32_t nr_col


cdef struct LayerC:
    MatrixC* W
    MatrixC* b
    int32_t nr_in
    int32_t nr_out
    void (*activate)(MatrixC* state) nogil
    void (*d_activate)(MatrixC* delta, const LayerC* layer, const MatrixC* state) nogil
    uint64_t id


cdef struct NetworkWeightsC:
    float* data
    int32_t nr_wide
    int32_t nr_deep
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
