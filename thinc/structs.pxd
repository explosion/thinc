from libc.stdint cimport int16_t, int, int32_t, uint64_t
from preshed.maps cimport MapStruct

from .typedefs cimport len_t, idx_t, atom_t


include "compile_time_constants.pxi"


ctypedef void (*do_update_t)(
    float* weights,
    float* momentum,
    float* gradient,
        len_t nr,
        const ConstantsC* hp
) nogil


ctypedef void (*do_feed_fwd_t)(
    float** fwd,
    float* averages,
        const float* W,
        const len_t* shape,
        int nr_below,
        int nr_above,
        const ConstantsC* hp
) nogil
 

ctypedef void (*do_feed_bwd_t)(
    float* G,
    float** bwd,
    float* averages,
        const float* W,
        const float* const* fwd,
        const len_t* shape,
        int nr_above,
        int nr_below,
        const ConstantsC* hp
) nogil


# Alias this, so that it matches our naming scheme
ctypedef MapStruct MapC


cdef struct ConstantsC:
    float a
    float b
    float c
    float d
    float e
    float g
    float h
    float i
    float j
    float k
    float l
    float m
    float n
    float o
    float p
    float q
    float r
    float s
    float t
    float u
    float w
    float x
    float y
    float z


cdef struct EmbedC:
    MapC** weights
    MapC** momentum
    idx_t* offsets
    len_t* lengths
    len_t nr


cdef struct NeuralNetC:
    do_feed_fwd_t feed_fwd
    do_feed_bwd_t feed_bwd
    do_update_t update

    len_t* widths
    float* weights
    float* gradient
    float* momentum

    float** averages
    
    EmbedC embed

    len_t nr_layer
    len_t nr_weight
    len_t nr_node

    ConstantsC hp


cdef struct ExampleC:
    int* is_valid
    float* costs
    uint64_t* atoms
    FeatureC* features
    float* scores

    float** fwd_state
    float** bwd_state
    int* widths

    int nr_class
    int nr_atom
    int nr_feat
    int nr_layer


cdef packed struct SparseArrayC:
    int32_t key
    float val


cdef struct FeatureC:
    int i
    uint64_t key
    float value


cdef struct SparseAverageC:
    SparseArrayC* curr
    SparseArrayC* avgs
    SparseArrayC* times


cdef struct TemplateC:
    int[MAX_TEMPLATE_LEN] indices
    int length
    atom_t[MAX_TEMPLATE_LEN] atoms
