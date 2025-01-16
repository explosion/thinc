from libcpp.memory cimport shared_ptr

ctypedef void (*sgemm_ptr)(bint transA, bint transB, int M, int N, int K,
                           float alpha, const float* A, int lda, const float* B,
                           int ldb, float beta, float* C, int ldc) nogil noexcept
ctypedef void (*dgemm_ptr)(bint transA, bint transB, int M, int N, int K,
                           double alpha, const double* A, int lda, const double* B,
                           int ldb, double beta, double* C, int ldc) nogil noexcept


ctypedef void (*saxpy_ptr)(int N, float alpha, const float* X, int incX,
                           float *Y, int incY) nogil noexcept


ctypedef void (*daxpy_ptr)(int N, double alpha, const double* X, int incX,
                           double *Y, int incY) nogil noexcept

ctypedef void (*sscal_ptr)(int N, float alpha, float* X, int incX) nogil
ctypedef void (*dscal_ptr)(int N, double alpha, double* X, int incX) nogil

# Forward-declaration of the BlasFuncs struct. This struct must be opaque, so
# that consumers of the CBlas class cannot become dependent on its size or
# ordering.
cdef struct BlasFuncs


cdef class CBlas:
    cdef shared_ptr[BlasFuncs] ptr


# Note: the following functions are intentionally standalone. If we make them
# methods of CBlas, Cython will generate and use a vtable. This makes it
# impossible to add new BLAS functions later without breaking the ABI.
#
# See https://github.com/explosion/thinc/pull/700 for more information.

cdef daxpy_ptr daxpy(CBlas cblas) nogil
cdef saxpy_ptr saxpy(CBlas cblas) nogil
cdef sgemm_ptr sgemm(CBlas cblas) nogil
cdef dgemm_ptr dgemm(CBlas cblas) nogil
cdef sscal_ptr sscal(CBlas cblas) nogil
cdef dscal_ptr dscal(CBlas cblas) nogil
cdef void set_daxpy(CBlas cblas, daxpy_ptr daxpy) nogil
cdef void set_saxpy(CBlas cblas, saxpy_ptr saxpy) nogil
cdef void set_sgemm(CBlas cblas, sgemm_ptr sgemm) nogil
cdef void set_dgemm(CBlas cblas, dgemm_ptr dgemm) nogil
cdef void set_sscal(CBlas cblas, sscal_ptr sscal) nogil
cdef void set_dscal(CBlas cblas, dscal_ptr dscal) nogil
