from libc.stdint cimport int16_t, int, int32_t, uint64_t
from preshed.maps cimport MapStruct

from .typedefs cimport len_t, idx_t, atom_t, weight_t


include "compile_time_constants.pxi"


ctypedef void (*do_update_t)(
    weight_t* weights,
    weight_t* momentum,
    weight_t* gradient,
        len_t nr,
        const ConstantsC* hp
) nogil


ctypedef void (*do_feed_fwd_t)(
    weight_t** fwd,
    weight_t* averages,
        const weight_t* W,
        const len_t* shape,
        int nr_below,
        int nr_above,
        const ConstantsC* hp
) nogil
 

ctypedef void (*do_feed_bwd_t)(
    weight_t* G,
    weight_t** bwd,
    weight_t* averages,
        const weight_t* W,
        const weight_t* const* fwd,
        const len_t* shape,
        int nr_above,
        int nr_below,
        const ConstantsC* hp
) nogil


# Alias this, so that it matches our naming scheme
ctypedef MapStruct MapC


cdef struct ConstantsC:
    weight_t a
    weight_t b
    weight_t c
    weight_t d
    weight_t e
    weight_t g
    weight_t h
    weight_t i
    weight_t j
    weight_t k
    weight_t l
    weight_t m
    weight_t n
    weight_t o
    weight_t p
    weight_t q
    weight_t r
    weight_t s
    weight_t t
    weight_t u
    weight_t w
    weight_t x
    weight_t y
    weight_t z


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
    weight_t* weights
    weight_t* gradient
    weight_t* momentum

    EmbedC embed

    len_t nr_layer
    len_t nr_weight
    len_t nr_node

    ConstantsC hp


cdef struct ExampleC:
    int* is_valid
    weight_t* costs
    uint64_t* atoms
    FeatureC* features
    weight_t* scores

    weight_t** fwd_state
    weight_t** bwd_state
    int* widths

    int nr_class
    int nr_atom
    int nr_feat
    int nr_layer


cdef packed struct SparseArrayC:
    int32_t key
    weight_t val


cdef struct FeatureC:
    int i
    uint64_t key
    weight_t value


cdef struct SparseAverageC:
    SparseArrayC* curr
    SparseArrayC* avgs
    SparseArrayC* times


cdef struct TemplateC:
    int[MAX_TEMPLATE_LEN] indices
    int length
    atom_t[MAX_TEMPLATE_LEN] atoms
