# cython: infer_types=True


cdef void parse_weights(
        int* W,
        int* bias,
            const len_t* widths, const int i, const int nr_layer) nogil:
    offset = 0
    for lyr in range(1, i):
        offset += widths[lyr] * widths[lyr-1] + widths[lyr]
    W[0] = offset
    bias[0] = W[0] + widths[i] * widths[i-1]


cdef void parse_batch_norm_weights(
        int* W,
        int* bias,
        int* gamma,
        int* beta,
        int* mean,
        int* variance,
            const len_t* widths,
            const int i, const int nr_layer) nogil:
    offset = 0
    for lyr in range(1, i):
        offset += widths[lyr] * widths[lyr-1] + widths[lyr] * 5
    W[0] = offset
    bias[0] = W[0] + widths[i] * widths[i-1]
    gamma[0] = bias[0] + widths[i]
    beta[0] = gamma[0] + widths[i]
    mean[0] = beta[0] + widths[i]
    variance[0] = mean[0] + widths[i]
