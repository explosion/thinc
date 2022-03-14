cdef void seq2col(float* output, const float* X, const int* L, int nW, int B, int I, int nL) nogil

cdef void backprop_seq2col(float* d_seqs,
        const float* d_cols, const int* L, int B, int I, int nW, int nL) nogil

cdef void cpu_maxout(float* best__bo, int* which__bo,
        const float* cands__bop, int B, int O, int P) nogil

cdef void cpu_backprop_maxout(float* dX__bop,
        const float* dX__bo, const int* which__bo, int B, int O, int P) nogil

cdef void cpu_reduce_mean(float* means__bo,
        const float* X__to, const int* lengths__b,
        int B, int T, int O) nogil

cdef void cpu_backprop_reduce_mean(float* dX__to,
        const float* d_means__bo, const int* lengths__b,
        int B, int T, int O) nogil

cdef void cpu_reduce_max(float* maxes__bo, int* which__bo,
        const float* X__to, const int* lengths__b,
        int B, int T, int O) nogil


cdef void cpu_backprop_reduce_max(float* dX__to,
        const float* d_maxes__bo, const int* which__bo, const int* lengths__b,
        int B, int T, int O) nogil
