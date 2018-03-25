cdef void simple_gemm(float* output, int o0, int o1,
                     const float* A, int a0, int a1,
                     const float* B, int b0, int b1,
                     int trans1, int trans2) nogil
 
#cdef void simple_ger(float* output, int o0, int o1,
#                     const float* A, int a0,
#                     const float* B, int b0) nogil
#
#cdef void scale(float* output, int o0, float scale) nogil

cdef void simple_axpy(float* output, int o0,
    const float* A, float scale) nogil
 
