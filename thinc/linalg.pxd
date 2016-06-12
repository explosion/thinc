# cython: profile=True
# cython: cdivision=True

cimport cython
from libc.stdint cimport int32_t
from libc.string cimport memcpy
from cymem.cymem cimport Pool

from .typedefs cimport weight_t

include "compile_time_constants.pxi"


# Copied from Shane Legg's Tokyo
cdef extern from "/opt/OpenBLAS/include/cblas.h":
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
    void cblas_dswap(int M, double  *x, int incX, double  *y, int incY) nogil
    void cblas_dscal(int N, double  alpha, double *x, int incX) nogil
    void cblas_dcopy(int N, double *x, int incX, double  *y, int incY) nogil
    void cblas_daxpy(int N, double  alpha, double *x, int incX, double *y, int incY ) nogil

    float cblas_sdot(int N, float  *x, int incX, float *y, int incY ) nogil
    float cblas_snrm2(int N, float  *x, int incX) nogil
    float cblas_sasum(int N, float  *x, int incX) nogil
    int cblas_isamax(int N, float  *x, int incX) nogil
    double cblas_ddot(int N, double  *x, int incX, double  *y, int incY ) nogil
    double cblas_dnrm2(int N, double  *x, int incX) nogil
    double cblas_dasum(int N, double  *x, int incX) nogil
    int cblas_idamax(int N, double  *x, int incX) nogil


    # BLAS level 2 routines
    void cblas_sgemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                    float  alpha, float *A, int lda, float *x, int incX,
                                 float beta, float *y, int incY) nogil

    void cblas_sger(CBLAS_ORDER Order, int M, int N, float  alpha, float  *x,
                                int incX, float  *y, int incY, float  *A, int lda) nogil

    void cblas_dgemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                     double  alpha, double  *A, int lda, double  *x, int incX,
                     double  beta, double  *y, int incY) nogil

    void cblas_dger(CBLAS_ORDER Order, int M, int N, double  alpha, double  *x,
                    int incX, double  *y, int incY, double  *A, int lda) nogil

    # BLAS level 3 routines
    void cblas_sgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 float  alpha, float  *A, int lda, float  *B, int ldb,
                                 float  beta, float  *C, int ldc) nogil
  
    void cblas_dgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 double  alpha, double  *A, int lda, double  *B, int ldb,
                                 double  beta, double  *C, int ldc) nogil


cdef extern from "math.h" nogil:
    weight_t exp(weight_t x)
    weight_t sqrt(weight_t x)


cdef class Matrix:
    cdef readonly Pool mem
    cdef weight_t* data
    cdef readonly int32_t nr_row
    cdef readonly int32_t nr_col


cdef class Vec:
    @staticmethod    
    cdef inline int arg_max(const weight_t* scores, const int n_classes) nogil:
        cdef int i
        cdef int best = 0
        cdef weight_t mode = scores[0]
        for i in range(1, n_classes):
            if scores[i] > mode:
                mode = scores[i]
                best = i
        return best

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
    cdef inline weight_t norm(const weight_t* vec, int32_t nr) nogil:
        cdef weight_t total = 0
        IF USE_BLAS:
            return cblas_dnrm2(nr, vec, 1)
        ELSE:
            for i in range(nr):
                total += vec[i] ** 2
            return sqrtf(total)

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
        IF USE_BLAS:
            cblas_dscal(nr, scal, vec, 1)
        ELSE:
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
            vec[i] = exp(vec[i])

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
        cdef int i
        IF USE_BLAS:
            cblas_daxpy(nr, scale, y, 1, x, 1)
        ELSE:
            for i in range(nr):
                x[i] += y[i] * scale
    
    @staticmethod
    cdef inline void batch_add_i(weight_t* x, 
                           const weight_t* y,
                           weight_t scale,
                           int32_t nr, int32_t nr_batch) nogil:
        # For fixed x, matrix of y
        cdef int i, _
        for _ in range(nr_batch):
            VecVec.add_i(x,
                y, scale, nr)
            y += nr
 
    @staticmethod
    cdef inline void add_pow(weight_t* output,
            const weight_t* x, const weight_t* y, weight_t power, int32_t nr) nogil:
        memcpy(output, x, sizeof(output[0]) * nr)
        VecVec.add_pow_i(output, y, power, nr)

   
    @staticmethod
    cdef inline void add_pow_i(weight_t* x, 
            const weight_t* y, weight_t power, int32_t nr) nogil:
        cdef int i
        for i in range(nr):
            x[i] += y[i] ** power
 
    @staticmethod
    cdef inline void mul(weight_t* output,
            const weight_t* x, const weight_t* y, int32_t nr) nogil:
        memcpy(output, x, sizeof(output[0]) * nr)
        VecVec.mul_i(output, y, nr)
   
    @staticmethod
    cdef inline void mul_i(weight_t* x, 
            const weight_t* y, int32_t nr) nogil:
        cdef int i
        for i in range(nr):
            x[i] *= y[i]

 
    @staticmethod
    cdef inline weight_t dot(
            const weight_t* x, const weight_t* y, int32_t nr) nogil:
        cdef int i
        cdef weight_t total = 0
        for i in range(nr):
            total += x[i] * y[i]
        return total
 
    @staticmethod
    cdef inline int arg_max_if_true(
            const weight_t* scores, const int* is_valid, const int n_classes) nogil:
        cdef int i
        cdef int best = -1
        for i in range(n_classes):
            if is_valid[i] and (best == -1 or scores[i] > scores[best]):
                best = i
        return best

    @staticmethod
    cdef inline int arg_max_if_zero(
            const weight_t* scores, const weight_t* costs, const int n_classes) nogil:
        cdef int i
        for i in range(n_classes):
            if costs[i] == 0 and (best == -1 or scores[i] > scores[best]):
                best = i
        return best


cdef class MatVec:
    @staticmethod
    cdef inline void add_i(weight_t* mat,
            const weight_t* vec, weight_t scale, int32_t nr_row, int32_t nr_col) nogil:
        cdef int i
        for i in range(nr_row):
            VecVec.add_i(mat + (i * nr_col),
                vec, 1.0, nr_col)

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
        cdef int i, row, col
        IF USE_BLAS:
            cblas_dgemv(
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
        ELSE:
            for i in range(nr_row):
                output[i] = 0
                row = i * nr_col
                for col in range(nr_col):
                    output[i] += mat[row + col] * vec[col]
    
    @staticmethod
    cdef inline void batch_dot(weight_t* output,
                         const weight_t* mat,
                         const weight_t* vec,
                         int32_t nr_row, int32_t nr_col, int32_t nr_batch) nogil:
        # Output dim: batch_size * nr_row
        # vec dim:    batch_size * nr_col
        # mat dim:    nr_row     * nr_col
        # batch_size must be M, because can't transpose C
        # so nr_row must be N
        # so nr_col must be K

        # vec:   M * K
        # mat.T: K * N
        # out:   M * N
        cdef int i, row, col
        IF USE_BLAS:
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                nr_batch,
                nr_row,
                nr_col,
                1.0,
                vec,
                nr_col,
                mat,
                nr_col,
                1.0,
                output,
                nr_row)
        ELSE:
            for b in range(nr_batch):
                MatVec.dot(output,
                    mat, vec, nr_row, nr_col)
                output += nr_col
                vec += nr_col

    @staticmethod
    cdef inline void T_dot(weight_t* output,
                             const weight_t* mat,
                             const weight_t* vec,
                             int32_t nr_row,
                             int32_t nr_col) nogil:
        cdef int i, row, col
        IF USE_BLAS:
            cblas_dgemv(CblasRowMajor, CblasTrans,
                nr_row, nr_col,
                1.0,
                mat, nr_col,
                vec, 1,
                0.0,
                output, 1
            )
        ELSE:
            for row in range(nr_row):
                for col in range(nr_col):
                    output[col] += vec[row] * mat[(row * nr_col) + col]

    @staticmethod
    cdef inline void batch_T_dot(weight_t* output,
                             const weight_t* mat,
                             const weight_t* vec,
                             int32_t nr_row,
                             int32_t nr_col,
                             int32_t nr_batch) nogil:
        cdef int _
        IF USE_BLAS:
            # output is (nr_batch, nr_col)
            # mat is (nr_row, nr_col)
            # vec is (nr_batch, nr_row)
            # Output defined as (M, N)
            # So
            # nr_batch = M
            # nr_col = N
            # nr_row = K
            #
            # vec:  M * K
            # mat:  K * N
            # out:  M * N
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                nr_batch,
                nr_col,
                nr_row,
                1.0,
                vec,
                nr_row,
                mat,
                nr_col,
                1.0,
                output,
                nr_col)
        ELSE:
            for _ in range(nr_batch):
                MatVec.T_dot(output,
                    mat, vec, nr_row, nr_col)
                output += nr_col
                vec += nr_row


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
        IF USE_BLAS:
            cblas_dger(
                CblasRowMajor,
                nr_row,
                nr_col,
                1.0,
                x,
                1,
                y,
                1,
                mat,
                nr_col)
        ELSE:
            for i in range(nr_row):
                row = i * nr_col
                for j in range(nr_col):
                    mat[row + j] += x[i] * y[j]

    @staticmethod 
    cdef inline void batch_add_outer_i(weight_t* output,
                                 const weight_t* x,
                                 const weight_t* y,
                                 int32_t nr_row,
                                 int32_t nr_col,
                                 int32_t nr_batch) nogil:
        # Output dim: nr_row * nr_col
        # x dim:    batch_size * nr_row
        # y dim:    batch_size * nr_col
        # 
        # Output is M*N (can't transpose)
        # nr_row = M
        # nr_col = N
        # batch_size = K

        # x.T:  M * K
        # y:    K * N
        # out:  M * N
        IF USE_BLAS:
            cblas_dgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                nr_row,
                nr_col,
                nr_batch,
                1.0,
                x,
                nr_row,
                y,
                nr_col,
                1.0,
                output,
                nr_col)
        ELSE:
            for _ in range(nr_batch):
                for i in range(nr_row):
                    row = i * nr_col
                    for j in range(nr_col):
                        mat[row + j] += x[i] * y[j]
                x += nr_row
                y += nr_col
