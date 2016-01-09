from libc.stdint cimport uint64_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uint16_t
from libc.stdint cimport int32_t



ctypedef float weight_t
ctypedef uint64_t atom_t
ctypedef uint64_t feat_t
ctypedef uint64_t hash_t
ctypedef int32_t class_t
ctypedef uint32_t count_t
ctypedef uint32_t time_t
ctypedef int32_t len_t
ctypedef int32_t idx_t


ctypedef void (*do_update_t)(
    float* weights,
    float* mtm,
    float* gradient,
        len_t nr,
        const void* _ext) nogil


ctypedef int (*do_iter_t)(
    void* _it,
        const int* widths,
            len_t nr_layer,
        int step_size
) nogil


ctypedef void (*do_feed_fwd_t)(
    float** fwd,
        const len_t* widths,
            len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const void* _it,
        const void* _ext
) nogil
 

ctypedef IteratorC (*do_begin_fwd_t)(
    float** fwd,
        const len_t* widths,
        len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const void* _ext
) nogil


ctypedef void (*do_end_fwd_t)(
    void* _it, float** fwd,
        const len_t* widths,
            len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const void* _ext
) nogil


ctypedef IteratorC (*do_begin_bwd_t)(
    float** bwd,
        const float* const* fwd,
        const len_t* widths,
        len_t nr_layer,
        const float* weights,
            const len_t nr_weight,
        const void* _ext
) nogil


ctypedef void (*do_feed_bwd_t)(
    float* bwd,
        const float* fwd,
        const len_t* widths,
            len_t nr_layer,
        const float* weights,
        len_t nr_weight,
        const void* _it,
        const void* _ext
) nogil


ctypedef void (*do_end_bwd_t)(
    void* _it, float* bwd,
        const float* fwd,
        const len_t* widths,
            len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const void* _ext
) nogil
