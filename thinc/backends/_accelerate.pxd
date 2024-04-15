cdef extern from "Accelerate/Accelerate.h":
    enum CBLAS_ORDER:     CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans
    enum CBLAS_UPLO:      CblasUpper, CblasLower
    enum CBLAS_DIAG:      CblasNonUnit, CblasUnit
    enum CBLAS_SIDE:      CblasLeft, CblasRight

    # BLAS level 1 routines

    void cblas_sswap(int M, float  *x, int incX, float  *y, int incY) nogil
    void cblas_sscal(int N, float  alpha, float  *x, int incX) nogil
    void cblas_scopy(int N, float  *x, int incX, float  *y, int incY) nogil
    void cblas_saxpy(int N, float  alpha, float  *x, int incX, float  *y, int incY ) nogil
    float cblas_sdot(int N, float  *x, int incX, float  *y, int incY ) nogil
    float cblas_snrm2(int N, float  *x, int incX) nogil
    float cblas_sasum(int N, float  *x, int incX) nogil
    int cblas_isamax(int N, float  *x, int incX) nogil

    # BLAS level 2 routines
    void cblas_sgemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                                 float  alpha, float  *A, int lda, float  *x, int incX,
                                 float  beta, float  *y, int incY) nogil

    void cblas_sger(CBLAS_ORDER Order, int M, int N, float  alpha, float  *x,
                                int incX, float  *y, int incY, float  *A, int lda) nogil

    # BLAS level 3 routines
    void cblas_sgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 float  alpha, float  *A, int lda, float  *B, int ldb,
                                 float  beta, float  *C, int ldc) nogil


cdef void sgemm(bint TransA, bint TransB, int M, int N, int K,
                    float alpha, const float* A, int lda, const float *B,
                    int ldb, float beta, float* C, int ldc) nogil


cdef void saxpy(int N, float alpha, const float* X, int incX,
                float *Y, int incY) nogil
