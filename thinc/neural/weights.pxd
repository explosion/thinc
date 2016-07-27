from ..typedefs cimport weight_t, len_t


cdef void parse_batch_norm_weights(
        int* W,
        int* bias,
        int* gamma,
        int* beta,
        int* mean,
        int* variance,
            const len_t* widths,
            const int i, const int nr_layer) nogil


cdef void parse_weights(
        int* W,
        int* bias,
            const len_t* widths,
            const int i, const int nr_layer) nogil
