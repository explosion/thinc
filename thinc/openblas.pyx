# cython: infer_types=True
# cython: cdivision=True
from .typedefs cimport weight_t


cdef extern from "cblas.h":
    enum CBLAS_ORDER:     CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans
    enum CBLAS_UPLO:      CblasUpper, CblasLower
    enum CBLAS_DIAG:      CblasNonUnit, CblasUnit
    enum CBLAS_SIDE:      CblasLeft, CblasRight

    # BLAS level 1 routines

    #void cblas_sswap(int M, float  *x, int incX, float  *y, int incY) nogil
    #void cblas_sscal(int N, float  alpha, float  *x, int incX) nogil
    #void cblas_scopy(int N, float  *x, int incX, float  *y, int incY) nogil
    void cblas_saxpy(int N, float  alpha, float  *x, int incX, float  *y, int incY ) nogil
    #void cblas_dswap(int M, double  *x, int incX, double  *y, int incY) nogil
    #void cblas_dscal(int N, double  alpha, double *x, int incX) nogil
    #void cblas_dcopy(int N, double *x, int incX, double  *y, int incY) nogil
    #void cblas_daxpy(int N, double  alpha, double *x, int incX, double *y, int incY ) nogil

    #float cblas_sdot(int N, float  *x, int incX, float *y, int incY ) nogil
    #float cblas_snrm2(int N, float  *x, int incX) nogil
    #float cblas_sasum(int N, float  *x, int incX) nogil
    #int cblas_isamax(int N, float  *x, int incX) nogil
    #double cblas_ddot(int N, double  *x, int incX, double  *y, int incY ) nogil
    #double cblas_dnrm2(int N, double  *x, int incX) nogil
    #double cblas_dasum(int N, double  *x, int incX) nogil
    #int cblas_idamax(int N, double  *x, int incX) nogil

    # BLAS level 2 routines
    #void cblas_sgemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
    #                float  alpha, float *A, int lda, float *x, int incX,
    #                             float beta, float *y, int incY) nogil

    #void cblas_sger(CBLAS_ORDER Order, int M, int N, float  alpha, float  *x,
    #                            int incX, float  *y, int incY, float  *A, int lda) nogil

    #void cblas_dgemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
    #                 double  alpha, double  *A, int lda, double  *x, int incX,
    #                 double  beta, double  *y, int incY) nogil

    #void cblas_dger(CBLAS_ORDER Order, int M, int N, double  alpha, double  *x,
    #                int incX, double  *y, int incY, double  *A, int lda) nogil

    # BLAS level 3 routines
    void cblas_sgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 float  alpha, float  *A, int lda, float  *B, int ldb,
                                 float  beta, float  *C, int ldc) nogil
  
    #void cblas_dgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
    #                             CBLAS_TRANSPOSE TransB, int M, int N, int K,
    #                             double  alpha, double  *A, int lda, double  *B, int ldb,
    #                             double  beta, double  *C, int ldc) nogil


cdef void simple_gemm(float* output, int o0, int o1,
                     const float* A, int a0, int a1,
                     const float* B, int b0, int b1,
                     int trans1, int trans2) nogil:
    cdef float alpha = 1.0
    cdef float beta = 1.0
    if not trans1 and not trans2:
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    o0, o1, b0, alpha, A, a1, B, b1, beta, output, o1)
    elif not trans1 and trans2:
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    o0, o1, b1, alpha, A, a1, B, b1, beta, output, o1)
    elif trans1 and trans2:
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                    o0, o1, b1, alpha, A, a1, B, b1, beta, output, o1)
    elif trans1 and not trans2:
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    o0, o1, b0,
                    alpha, A, a1, B, b1,
                    beta, output, o1)
    else:
        with gil:
            dims = '(%d, %d) = (%d, %d) @ (%d, %d)' % (o0, o1, a0, a1, b0, b1)
            raise ValueError("Invalid dimensions for GEMM: %s" % dims)
 

cdef void simple_ger(float* output, int o0, int o1,
                     const float* A, int a0,
                     const float* B, int b0) nogil:
    cdef float alpha = 1.
    pass
    #cblas_sger(CblasRowMajor, o0, o1, alpha, A, 1, B, 1,
    #    output, o1)


#cdef void scale(float* output, int o0, float scale) nogil:
#    cblas_sscal(o0, scale, output, 1)


cdef void simple_axpy(float* output, int o0,
        const float* A, float scale) nogil:

    cblas_saxpy(o0, 1., A, 1, output, 1)
