from libc.stdint cimport int16_t, int32_t, uint64_t
from preshed.maps cimport MapStruct
from .typedefs cimport weight_t, atom_t


include "compile_time_constants.pxi"

# Alias this, so that it matches our naming scheme
ctypedef MapStruct MapC


cdef struct NeuralNetC:
    int* widths
    weight_t* weights
    MapC* sparse
    void* opt

    void (*update)(void* opt, weight_t* weights, weight_t* gradient,
        int nr_weight, const int* widths, int nr_layer)

    int32_t nr_layer
    int32_t nr_weight

    weight_t eps
    weight_t rho
    weight_t eta


cdef struct ExampleC:
    int* is_valid
    weight_t* costs
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


cdef struct BatchC:
    ExampleC* egs
    weight_t* gradient
    MapC* sparse
    int nr_eg


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
