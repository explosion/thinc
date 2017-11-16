# cython: infer_types=True
# cython: cdivision=True

cimport cython
from libc.stdint cimport int32_t
from libc.string cimport memset, memcpy
from cymem.cymem cimport Pool


from .typedefs cimport weight_t

include "compile_time_constants.pxi"

IF USE_BLAS:
    from blis cimport cy as blis

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
        if n_classes == 2:
            return 0 if scores[0] > scores[1] else 1
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
        for i in range(nr):
            total += vec[i] ** 2
        return sqrt(total)

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
    cdef inline void mul_i(weight_t* vec, weight_t scal, int32_t nr) nogil:
        cdef int i
        IF USE_BLAS:
            blis.scalv(blis.NO_CONJUGATE, nr, scal, vec, 1)
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
            blis.axpyv(blis.NO_CONJUGATE, nr, scale, <weight_t*>y, 1, x, 1)
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
        cdef int best = -1
        for i in range(n_classes):
            if costs[i] == 0 and (best == -1 or scores[i] > scores[best]):
                best = i
        return best


cdef class Mat:
    @staticmethod
    cdef inline void mean_row(weight_t* Ex,
            const weight_t* mat, int32_t nr_row, int32_t nr_col) nogil:
        memset(Ex, 0, sizeof(Ex[0]) * nr_col)
        for i in range(nr_row):
            VecVec.add_i(Ex, &mat[i * nr_col], 1.0, nr_col)
        Vec.mul_i(Ex, 1.0 / nr_row, nr_col)

    @staticmethod
    cdef inline void var_row(weight_t* Vx,
            const weight_t* mat, const weight_t* Ex,
            int32_t nr_row, int32_t nr_col, weight_t eps) nogil:
        # From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        if nr_row == 0 or nr_col == 0:
            return
        cdef weight_t sum_, sum2
        for i in range(nr_col):
            sum_ = 0.0
            sum2 = 0.0
            for j in range(nr_row):
                x = mat[j * nr_col + i]
                sum2 += (x - Ex[i]) ** 2
                sum_ += x - Ex[i]
            Vx[i] = (sum2 - sum_**2 / nr_row) / nr_row
            Vx[i] += eps
 

cdef class MatVec:
    @staticmethod
    cdef inline void add_i(weight_t* mat,
            const weight_t* vec, weight_t scale, int32_t nr_row, int32_t nr_col) nogil:
        cdef int i
        for i in range(nr_row):
            VecVec.add_i(mat + (i * nr_col),
                vec, scale, nr_col)

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
        cdef double zero = 0.0
        IF USE_BLAS:
            blis.gemv(
                blis.NO_TRANSPOSE,
                blis.NO_CONJUGATE,
                nr_row,
                nr_col,
                1.0,
                <weight_t*>mat, nr_col, 1,
                <weight_t*>vec, 1,
                1.0,
                output, 1
            )
        ELSE:
            for i in range(nr_row):
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
        cdef double one = 1.0
        IF USE_BLAS:
            blis.gemm(
                blis.NO_TRANSPOSE,
                blis.TRANSPOSE,
                nr_batch,
                nr_row,
                nr_col,
                1.0,
                <weight_t*>vec,
                nr_col,
                1,
                <weight_t*>mat,
                nr_col,
                1,
                1.0,
                output,
                nr_row,
                1)
        ELSE:
            for b in range(nr_batch):
                MatVec.dot(output,
                    mat, vec, nr_row, nr_col)
                output += nr_row
                vec += nr_col

    @staticmethod
    cdef inline void T_dot(weight_t* output,
                             const weight_t* mat,
                             const weight_t* vec,
                             int32_t nr_row,
                             int32_t nr_col) nogil:
        cdef int i, row, col
        cdef double zero = 0.0
        cdef double one = 1.0
        IF USE_BLAS:
            blis.gemv(
                blis.TRANSPOSE,
                blis.NO_CONJUGATE,
                nr_row, nr_col,
                1.0,
                <weight_t*>mat, nr_col, 1,
                <weight_t*>vec, 1,
                1.0,
                output, 1,
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
        cdef double one = 1.0
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
            blis.gemm(
                blis.NO_TRANSPOSE,
                blis.NO_TRANSPOSE,
                nr_batch,
                nr_col,
                nr_row,
                1.0,
                <weight_t*>vec,
                nr_row,
                1,
                <weight_t*>mat,
                nr_col,
                1,
                1.0,
                output,
                nr_col,
                1)
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
        cdef double one = 1.0
        IF USE_BLAS:
            blis.ger(
                blis.NO_CONJUGATE, blis.NO_CONJUGATE,
                nr_row, nr_col,
                1.0,
                <weight_t*>x, 1,
                <weight_t*>y, 1,
                mat, nr_col, 1
            )
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
        cdef double one = 1.0
        IF USE_BLAS:
            blis.gemm(
                blis.TRANSPOSE,
                blis.NO_TRANSPOSE,
                nr_row,
                nr_col,
                nr_batch,
                1.0,
                <weight_t*>x,
                nr_row,
                1,
                <weight_t*>y,
                nr_col,
                1,
                1.0,
                output,
                nr_col,
                1)
        ELSE:
            for _ in range(nr_batch):
                for i in range(nr_row):
                    row = i * nr_col
                    for j in range(nr_col):
                        output[row + j] += x[i] * y[j]
                x += nr_row
                y += nr_col
