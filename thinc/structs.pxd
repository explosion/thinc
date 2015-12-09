from libc.stdint cimport int16_t, int32_t, uint64_t
from .typedefs cimport weight_t, atom_t


include "compile_time_constants.pxi"


cdef struct ExampleC:
    int* is_valid
    int* costs
    atom_t* atoms
    FeatureC* features
    weight_t* scores

    weight_t* gradient
    
    weight_t** fwd_state
    weight_t** bwd_state

    int nr_class
    int nr_atom
    int nr_feat
    
    int guess
    int best
    int cost


cdef struct LayerC:
    void (*forward)(
        weight_t* activity,
        const weight_t* W,
        const weight_t* input_, 
        const weight_t* bias,
        int32_t nr_out,
        int32_t nr_wide
    ) nogil

    void (*backward)(
        weight_t* delta_out,
        const weight_t* delta_in,
        const weight_t* signal_out,
        const weight_t* W,
        int32_t nr_wide, 
        int32_t nr_out
    ) nogil

    int32_t nr_wide
    int32_t nr_out

    int32_t W
    int32_t bias


cdef struct HyperParamsC:
    weight_t alpha
    weight_t beta
    weight_t gamma
    weight_t eta
    weight_t epsilon
    weight_t rho
    weight_t sigma
    weight_t tau


cdef struct SparseArrayC:
    int32_t key
    weight_t val


cdef struct FeatureC:
    int32_t i
    int32_t length
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
