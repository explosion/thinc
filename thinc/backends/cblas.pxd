from libcpp.memory cimport shared_ptr


ctypedef void (*sgemm_ptr)(bint transA, bint transB, int M, int N, int K,
                           float alpha, const float* A, int lda, const float *B,
                           int ldb, float beta, float* C, int ldc) nogil


ctypedef void (*saxpy_ptr)(int N, float alpha, const float* X, int incX,
                           float *Y, int incY) nogil


# Forward-declaration of the BlasFuncs struct. This struct must be opaque, so
# that consumers of the CBlas class cannot become dependent on its size or
# ordering.
cdef struct BlasFuncs


cdef class CBlas:
    cdef shared_ptr[BlasFuncs] ptr
    cdef saxpy_ptr saxpy(self) nogil
    cdef sgemm_ptr sgemm(self) nogil
    cdef void set_saxpy(self, saxpy_ptr saxpy) nogil
    cdef void set_sgemm(self, sgemm_ptr sgemm) nogil
