# cython: profile=True

cimport cython
from libc.stdint cimport int32_t
from libc.string cimport memcpy
from libc.math cimport isnan as c_is_nan
from cymem.cymem cimport Pool

from .typedefs cimport weight_t


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


cdef extern from "math.h" nogil:
    float expf(float x)


cdef class Matrix:
    cdef readonly Pool mem
    cdef weight_t* data
    cdef readonly int32_t nr_row
    cdef readonly int32_t nr_col


cdef class Vec:
    @staticmethod
    cdef inline int has_nan(const weight_t* weights, int n) nogil:
        cdef int i
        for i in range(n):
            if c_is_nan(weights[i]):
                return 1
        else:
            return 0

    @staticmethod
    cdef inline weight_t max(const weight_t* x, int32_t nr) nogil:
        if nr == 0:
            return 0
        cdef int i
        cdef weight_t mode = x[0]
        for i in range(1, nr):
            if x[i] > mode:
                mode = x[i]
        return mode

    @staticmethod
    cdef inline weight_t sum(const weight_t* vec, int32_t nr) nogil:
        cdef int i
        cdef weight_t total = 0
        for i in range(nr):
            total += vec[i]
        return total

    @staticmethod
    cdef inline void add(weight_t* output, const weight_t* x,
            weight_t inc, int32_t nr) nogil:
        memcpy(output, x, sizeof(output[0]) * nr)
        Vec.add_i(output, inc, nr)

    @staticmethod
    cdef inline void add_i(weight_t* vec, weight_t inc, int32_t nr) nogil:
        cdef int i
        for i in range(nr):
            vec[i] += inc

    @staticmethod
    cdef inline void mul(weight_t* output, const weight_t* vec, weight_t scal,
            int32_t nr) nogil:
        memcpy(output, vec, sizeof(output[0]) * nr)
        Vec.mul_i(output, scal, nr)

    @staticmethod
    cdef inline void mul_i(weight_t* vec, const weight_t scal, int32_t nr) nogil:
        cdef int i
        for i in range(nr):
            vec[i] *= scal

    @staticmethod
    cdef inline void pow(weight_t* output, const weight_t* vec, weight_t scal,
            int32_t nr) nogil:
        memcpy(output, vec, sizeof(output[0]) * nr)
        Vec.pow_i(output, scal, nr)

    @staticmethod
    cdef inline void pow_i(weight_t* vec, const weight_t scal, int32_t nr) nogil:
        cdef int i
        for i in range(nr):
            vec[i] **= scal

    @staticmethod
    @cython.cdivision(True)
    cdef inline void div(weight_t* output, const weight_t* vec, weight_t scal,
            int32_t nr) nogil:
        memcpy(output, vec, sizeof(output[0]) * nr)
        Vec.div_i(output, scal, nr)

    @staticmethod
    @cython.cdivision(True)
    cdef inline void div_i(weight_t* vec, const weight_t scal, int32_t nr) nogil:
        cdef int i
        for i in range(nr):
            vec[i] /= scal

    @staticmethod
    cdef inline void exp(weight_t* output, const weight_t* vec, int32_t nr) nogil:
        memcpy(output, vec, sizeof(output[0]) * nr)
        Vec.exp_i(output, nr)

    @staticmethod
    cdef inline void exp_i(weight_t* vec, int32_t nr) nogil:
        cdef int i
        for i in range(nr):
            vec[i] = expf(vec[i])

    @staticmethod
    cdef inline void reciprocal_i(weight_t* vec, int32_t nr) nogil:
        cdef int i
        for i in range(nr):
            vec[i] = 1.0 / vec[i]


cdef class VecVec:
    @staticmethod
    cdef inline void add(weight_t* output,
                         const weight_t* x, 
                         const weight_t* y,
                         weight_t scale,
                         int32_t nr) nogil:
        memcpy(output, x, sizeof(output[0]) * nr)
        VecVec.add_i(output, y, scale, nr)
   
    @staticmethod
    cdef inline void add_i(weight_t* x, 
                           const weight_t* y,
                           weight_t scale,
                           int32_t nr) nogil:
        #cblas_saxpy(nr, scale, x, 1, y, 1)
        cdef int i
        for i in range(nr):
            x[i] += y[i] * scale
 
    @staticmethod
    cdef inline void add_pow(weight_t* output,
                         const weight_t* x, 
                         const weight_t* y,
                         weight_t power,
                         int32_t nr) nogil:
        memcpy(output, x, sizeof(output[0]) * nr)
        VecVec.add_pow_i(output, y, power, nr)

   
    @staticmethod
    cdef inline void add_pow_i(weight_t* x, 
                               const weight_t* y,
                               weight_t power,
                               int32_t nr) nogil:
        cdef int i
        for i in range(nr):
            x[i] += y[i] ** power
 
    @staticmethod
    cdef inline void mul(weight_t* output,
                         const weight_t* x, 
                         const weight_t* y,
                         int32_t nr) nogil:
        memcpy(output, x, sizeof(output[0]) * nr)
        VecVec.mul_i(output, y, nr)
   
    @staticmethod
    cdef inline void mul_i(weight_t* x, 
                           const weight_t* y,
                           int32_t nr) nogil:
        cdef int i
        for i in range(nr):
            x[i] *= y[i]

 
    @staticmethod
    cdef inline weight_t dot(const weight_t* x, 
                             const weight_t* y,
                             int32_t nr) nogil:
        cdef int i
        cdef weight_t total = 0
        for i in range(nr):
            total += x[i] * y[i]
        return total
 

cdef class MatVec:
    @staticmethod
    cdef inline void mul(weight_t* output,
                         const weight_t* mat,
                         const weight_t* vec,
                         int32_t nr_row, int32_t nr_col) nogil:
        memcpy(output, mat, sizeof(output[0]) * nr_row * nr_col)
        MatVec.mul_i(output, vec, nr_row, nr_col)

    @staticmethod
    cdef inline void mul_i(weight_t* mat,
                           const weight_t* vec,
                           int32_t nr_row, int32_t nr_col) nogil:
        cdef int i, row, col
        for i in range(nr_row):
            row = i * nr_col
            for col in range(nr_col):
                mat[row + col] *= vec[col]

    @staticmethod
    cdef inline void dot(weight_t* output,
                         const weight_t* mat,
                         const weight_t* vec,
                         int32_t nr_row, int32_t nr_col) nogil:
        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            nr_row,
            nr_col,
            1.0,
            mat,
            nr_col,
            vec,
            1,
            0.0,
            output,
            1
        )
        #cdef int i, row, col
        #for i in range(nr_row):
        #    output[i] = 0
        #    row = i * nr_col
        #    for col in range(nr_col):
        #        output[i] += mat[row + col] * vec[col]

    @staticmethod
    cdef inline void T_dot(weight_t* output,
                             const weight_t* mat,
                             const weight_t* vec,
                             int32_t nr_row,
                             int32_t nr_col) nogil:
        cblas_sgemv(
            CblasRowMajor,
            CblasTrans,
            nr_row,
            nr_col,
            1.0,
            mat,
            nr_col,
            vec,
            1,
            0.0,
            output,
            1
        )
        #cdef int i, row, col
        #for row in range(nr_row):
        #    for col in range(nr_col):
        #        output[col] += vec[row] * mat[(row * nr_col) + col]


cdef class MatMat:
    @staticmethod
    cdef inline void add(weight_t* output,
                         const weight_t* x,
                         const weight_t* y,
                         int32_t nr_row, int32_t nr_col) nogil:
        memcpy(output, x, sizeof(output[0]) * nr_row * nr_col)
        MatMat.add_i(output, y, nr_row, nr_col)

    @staticmethod
    cdef inline void add_i(weight_t* x,
                           const weight_t* y,
                           int32_t nr_row, int32_t nr_col) nogil:
        cdef int i, row, col
        for i in range(nr_row):
            row = i * nr_col
            for col in range(nr_col):
                x[row + col] += y[row + col]

    @staticmethod
    cdef inline void mul(weight_t* output,
                         const weight_t* x,
                         const weight_t* y,
                         int32_t nr_row, int32_t nr_col) nogil:
        memcpy(output, x, sizeof(output[0]) * nr_row * nr_col)
        MatMat.mul_i(output, y, nr_row, nr_col)

    @staticmethod
    cdef inline void mul_i(weight_t* x,
                           const weight_t* y,
                           int32_t nr_row, int32_t nr_col) nogil:
        cdef int i, row, col
        for i in range(nr_row):
            row = i * nr_col
            for col in range(nr_col):
                x[row + col] *= y[row + col]

    @staticmethod 
    cdef inline void add_outer_i(weight_t* mat,
                                 const weight_t* x,
                                 const weight_t* y,
                                 int32_t nr_row,
                                 int32_t nr_col) nogil:
        cdef int i, j, row
        for i in range(nr_row):
            row = i * nr_col
            for j in range(nr_col):
                mat[row + j] += x[i] * y[j]
