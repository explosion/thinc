from cymem.cymem cimport Pool

from .structs cimport MatrixC

# Copied from Shane Legg's Tokyo
cdef extern from "cblas.h":

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
    float  cblas_snrm2(int N, float  *x, int incX) nogil
    float  cblas_sasum(int N, float  *x, int incX) nogil
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


cdef enum ReturnCode:
    ReturnSuccess = 0
    ReturnMismatchX
    ReturnMismatchY


cdef inline ReturnCode check_data_integrity(const MatrixC* me, const MatrixC* you) nogil:
    return ReturnSuccess


cdef class Matrix:
    cdef Pool mem
    cdef MatrixC* c

    @staticmethod
    cdef inline MatrixC* initC(Pool mem, int nr_row, int nr_col) except NULL:
        c = <MatrixC*>mem.alloc(1, sizeof(MatrixC))
        c.nr_row = nr_row
        c.nr_col = nr_col
        c.data = <float*>mem.alloc(nr_row * nr_col, sizeof(float))
        return c

    @staticmethod
    cdef inline float getC(const MatrixC* me, int row, int col) nogil:
        cdef int row_stride = me.nr_col
        cdef int col_stride = 1
        return me.data[(row * row_stride) + (col * col_stride)]

    @staticmethod
    cdef inline void setC(MatrixC* me, int row, int col, float value) nogil:
        cdef int row_stride = me.nr_col
        cdef int col_stride = 1
        me.data[(row * row_stride) + (col * col_stride)] = value

    @staticmethod
    cdef inline int dot_biasC(MatrixC* dest, const MatrixC* x, const MatrixC* W,
                              const MatrixC* b) nogil:
        # Use simple implementation first, before we fuss with BLAS
        cdef int i, j
        cdef float value, total
        for i in range(x.nr_col):
            value = Matrix.getC(x, 0, i)
            total = Matrix.getC(b, 0, i)
            for j in range(W.nr_col):
                total += value * Matrix.getC(W, i, j)
            Matrix.setC(dest, 0, i, total)
        #Matrix.imulC(me, W)
        #Matrix.iaddC(me, b, 1.0)

    @staticmethod
    cdef inline int iaddC(MatrixC* me, const MatrixC* you, float scale) nogil:
        cblas_saxpy(you.nr_col, scale, you.data, 1, me.data, 1)

    @staticmethod
    cdef inline int imulC(MatrixC* me, const MatrixC* A) nogil:
        cblas_sgemv(
            CblasRowMajor, CblasNoTrans,
            A.nr_row, A.nr_col,
            1.0, A.data,
            A.nr_col,
            me.data, 1, 0.0,
            me.data, 1
        )
