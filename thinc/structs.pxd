from libc.stdint cimport int16_t, int, uint64_t
from preshed.maps cimport MapStruct


include "compile_time_constants.pxi"

# Alias this, so that it matches our naming scheme
ctypedef MapStruct MapC


cdef struct EmbeddingC:
    MapC** tables
    float** defaults
    int* offsets
    int* lengths
    int nr


cdef struct NeuralNetC:
    do_iter_t iterate
    do_begin_fwd_t begin_fwd
    do_end_fwd_t end_fwd
    do_feed_fwd_t feed_fwd
    do_begin_bwd_t begin_bwd
    do_feed_bwd_t feed_bwd
    do_end_bwd_t end_bwd
    do_update_t update

    int* widths
    float* weights
    float* gradient
    float* momentum
    float* averages
    
    MapC** sparse_weights
    MapC** sparse_gradient
    MapC** sparse_momentum
    MapC** sparse_averages

    int* embed_offsets
    int* embed_lengths
    float** embed_defaults

    int nr_layer
    int nr_weight
    int nr_embed

    float alpha
    float eta
    float rho
    float eps


cdef struct ExampleC:
    int* is_valid
    float* costs
    uint64_t* atoms
    FeatureC* features
    float* scores

    float* fine_tune
    
    float** fwd_state
    float** bwd_state

    int nr_class
    int nr_atom
    int nr_feat
    
    int guess
    int best
    int cost


# Iteration controller
cdef struct IteratorC:
    int nr_out
    int nr_in
    int i
    int W
    int bias
    int gamma
    int below
    int here
    int above
    int Ex
    int Vx
    int E_dXh
    int E_dXh_Xh


cdef struct SparseArrayC:
    int key
    float val


cdef struct FeatureC:
    int i
    uint64_t key
    float value


#cdef struct SparseAverageC:
#    SparseArrayC* curr
#    SparseArrayC* avgs
#    SparseArrayC* times
#
#
#cdef struct TemplateC:
#    int[MAX_TEMPLATE_LEN] indices
#    int length
#    atom_t[MAX_TEMPLATE_LEN] atoms
