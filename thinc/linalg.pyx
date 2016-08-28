# cython: infer_types=True
# cython: cdivision=True

include "compile_time_constants.pxi"


cdef void v_fill(weights_ft vec, weight_t value, int nr) nogil:
    if weights_ft is dense_weights_t:
        for i in range(nr):
            vec[i] = value
    else:
        for i in range(nr):
            x = vec[i]
            while x.key >= 0:
                x.val = value
                x += 1


cdef weight_t v_norm(const_weights_ft vec, int32_t nr) nogil:
    cdef weight_t total = 0
    if const_weights_ft is const_dense_weights_t:
        for i in range(nr):
            total += vec[i] ** 2
    else:
        for i in range(nr):
            x = vec[i]
            while x.key >= 0:
                total += x.val ** 2
                x += 1
    return sqrt(total)

cdef void v_mul(weights_ft vec, weight_t scal, int32_t nr) nogil:
    cdef int i
    if weights_ft is dense_weights_t:
        IF USE_BLAS:
            blis.scalv(blis.NO_CONJUGATE, nr, scal, vec, 1)
        ELSE:
            for i in range(nr):
                vec[i] *= scal
    else:
        for i in range(nr):
            x = vec[i]
            while x.key >= 0:
                x.val *= scal
                x += 1

cdef void v_pow(weights_ft vec, const weight_t scal, int32_t nr) nogil:
    cdef int i
    if weights_ft is dense_weights_t:
        for i in range(nr):
            vec[i] **= scal
    else:
        for i in range(nr):
            x = vec[i]
            while x.key >= 0:
                x.val **= scal
                x += 1

cdef void vv_add(weights_ft x, 
                    const_weights_ft y,
                    weight_t scale,
                    int32_t nr) nogil:
    cdef int i
    if weights_ft is dense_weights_t and const_weights_ft is const_dense_weights_t:
        IF USE_BLAS:
            blis.axpyv(blis.NO_CONJUGATE, nr, scale, <weight_t*>y, 1, x, 1)
        ELSE:
            for i in range(nr):
                x[i] += y[i] * scale
    elif weights_ft is sparse_weights_t and const_weights_ft is const_sparse_weights_t:
        for i in range(nr):
            x_i = x[i]
            y_i = y[i]
            while x_i.key >= 0:
                x_i.val += y_i.val * scale
                x_i += 1
                y_i += 1
    else:
        # TODO: Panic
        pass

cdef void vv_batch_add(weight_t* x, 
                       const weight_t* y,
                       weight_t scale,
                       int32_t nr, int32_t nr_batch) nogil:
    # For fixed x, matrix of y
    cdef int i, _
    for _ in range(nr_batch):
        VecVec.add_i(x,
            y, scale, nr)
        y += nr


cdef void vv_add_pow(weights_ft x, 
                    const_weights_ft y, weight_t power, int32_t nr) nogil:
    cdef int i
    if weights_ft is dense_weights_t and const_weights_ft is const_dense_weights_t:
        for i in range(nr):
            x[i] += y[i] ** power
    elif weights_ft is sparse_weights_t and const_weights_ft is const_sparse_weights_t:
        pass
    else:
        # TODO: Panic
        pass


cdef void vv_mul(weights_ft x, const_weights_ft y, int32_t nr) nogil:
    cdef int i
    if weights_ft is dense_weights_t and const_weights_ft is const_dense_weights_t:
        for i in range(nr):
            x[i] *= y[i]
    elif weights_ft is sparse_weights_t and const_weights_ft is const_sparse_weights_t:
        for i in range(nr):
            x_i = x[i]
            y_i = y[i]
            while x_i.key >= 0:
                x_i.val *= y_i.val
                x_i += 1
                y_i += 1
    else:
        # TODO: Panic
        pass


cdef weight_t vv_dot(const weight_t* x, const weight_t* y, int32_t nr) nogil:
    cdef int i
    cdef weight_t total = 0
    for i in range(nr):
        total += x[i] * y[i]
    return total


cdef int arg_max_if_true(const weight_t* scores, const int* is_valid, const int n_classes) nogil:
    cdef int i
    cdef int best = -1
    for i in range(n_classes):
        if is_valid[i] and (best == -1 or scores[i] > scores[best]):
            best = i
    return best

cdef int arg_max_if_zero(
        const weight_t* scores, const weight_t* costs, const int n_classes) nogil:
    cdef int i
    cdef int best = -1
    for i in range(n_classes):
        if costs[i] == 0 and (best == -1 or scores[i] > scores[best]):
            best = i
    return best


cdef void mv_add(weight_t* mat,
        const weight_t* vec, weight_t scale, int32_t nr_row, int32_t nr_col) nogil:
    cdef int i
    for i in range(nr_row):
        vv_add(mat + (i * nr_col),
            vec, scale, nr_col)

cdef void mv_mul(weight_t* mat,
              const weight_t* vec,
              int32_t nr_row, int32_t nr_col) nogil:
    cdef int i, row, col
    for i in range(nr_row):
        row = i * nr_col
        for col in range(nr_col):
            mat[row + col] *= vec[col]

cdef void mv_dot(weight_t* output,
                    const weight_t* mat,
                    const weight_t* vec,
                    int32_t nr_row, int32_t nr_col) nogil:
    cdef int i, row, col
    cdef double zero = 0.0
    cdef double one = 1.0
    if weights_ft is dense_weights_t and const_weights_ft is const_dense_weights_t:
        IF True:
            blis.gemv(
                blis.NO_TRANSPOSE,
                blis.NO_CONJUGATE,
                nr_row,
                nr_col,
                one,
                <weight_t*>mat, nr_col, 1,
                <weight_t*>vec, 1,
                one,
                output, 1
            )
        ELSE:
            for i in range(nr_row):
                row = i * nr_col
                for col in range(nr_col):
                    output[i] += mat[row + col] * vec[col]
    else:
        pass
    
cdef void mv_batch_dot(weight_t* output,
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
    if weights_ft is dense_weights_t and const_weights_ft is const_dense_weights_t:
        IF True:
            blis.gemm(
                blis.NO_TRANSPOSE,
                blis.TRANSPOSE,
                nr_batch,
                nr_row,
                nr_col,
                one,
                <weight_t*>vec,
                nr_col,
                1,
                <weight_t*>mat,
                nr_col,
                1,
                one,
                output,
                nr_row,
                1)
        ELSE:
            for b in range(nr_batch):
                MatVec.dot(output,
                    mat, vec, nr_row, nr_col)
                output += nr_row
                vec += nr_col
    else:
        pass

cdef void mv_T_dot(weight_t* output,
        const weight_t* mat,
        const weight_t* vec,
        int32_t nr_row,
        int32_t nr_col) nogil:
    cdef int i, row, col
    cdef double zero = 0.0
    cdef double one = 1.0
    if weights_ft is dense_weights_t and const_weights_ft is const_dense_weights_t:
        IF True:
            blis.gemv(
                blis.TRANSPOSE,
                blis.NO_CONJUGATE,
                nr_row, nr_col,
                one,
                <weight_t*>mat, nr_col, 1,
                <weight_t*>vec, 1,
                one,
                output, 1,
            )
        ELSE:
            for row in range(nr_row):
                for col in range(nr_col):
                    output[col] += vec[row] * mat[(row * nr_col) + col]
    elif weights_ft is sparse_weights_t and const_weights_ft is const_sparse_weights_ft:
        for row in range(nr_row):
            for col in range(nr_col):
                output[col] += vec[row] * mat[(row * nr_col) + col]


cdef void mv_batch_T_dot(weight_t* output,
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
            one,
            <weight_t*>vec,
            nr_row,
            1,
            <weight_t*>mat,
            nr_col,
            1,
            one,
            output,
            nr_col,
            1)
    ELSE:
        for _ in range(nr_batch):
            MatVec.T_dot(output,
                mat, vec, nr_row, nr_col)
            output += nr_col
            vec += nr_row


cdef void mm_add(weights_ft x,
        const_weights_ft y, int32_t nr_row, int32_t nr_col) nogil:
    cdef int i, row, col
    if weights_ft is dense_weights_t and const_weights_ft is const_denst_weights_t:
        for i in range(nr_row):
            row = i * nr_col
            for col in range(nr_col):
                x[row + col] += y[row + col]
    elif weights_ft is sparse_weights_t and const_weights_ft is const_sparse_weights_ft:
        for i in range(nr_row):
            x_i = x[i]
            y_i = y[i]
            while x_i.key >= 0:
                x_i.val *= y_i.val
                x_i += 1
                y_i += 1


cdef void mm_mul(weights_ft x,
        const_weights_ft y, int32_t nr_row, int32_t nr_col) nogil:
    cdef int i, row, col
    if weights_ft is dense_weights_t and const_weights_ft is const_denst_weights_t:
        for i in range(nr_row):
            row = i * nr_col
            for col in range(nr_col):
                x[row + col] *= y[row + col]
    elif weights_ft is sparse_weights_t and const_weights_ft is const_sparse_weights_ft:
        for i in range(nr_row):
            x_i = x[i]
            y_i = y[i]
            while x_i.key >= 0:
                x_i.val *= y_i.val
                x_i *= 1
                y_i *= 1


cdef void mm_add_outer(weights_ft mat,
                             const_weights_ft x,
                             const_weights_ft y,
                             int32_t nr_row,
                             int32_t nr_col) nogil:
    cdef int i, j, row
    cdef double one = 1.0
    if weights_ft is dense_weights_t and const_weights_ft is const_denst_weights_t:
        IF True:
            blis.ger(
                blis.NO_CONJUGATE, blis.NO_CONJUGATE,
                nr_row, nr_col,
                one,
                <weight_t*>x, 1,
                <weight_t*>y, 1,
                mat, nr_col, 1
            )
        ELSE:
            for i in range(nr_row):
                row = i * nr_col
                for j in range(nr_col):
                    mat[row + j] += x[i] * y[j]
    elif weights_ft is sparse_weights_t and const_weights_ft is const_sparse_weights_ft:
        for i in range(nr_row):
            cell = mat[i]
            while cell.key >= 0:
                cell.val += x[i] * y[cell.key]
                cell += 1
 


cdef void mm_batch_add_outer(weights_ft* output,
                             const_weights_ft x,
                             const_weights_ft y,
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
    if weights_ft is dense_weights_t and const_weights_ft is const_denst_weights_t:
        IF True:
            blis.gemm(
                blis.TRANSPOSE,
                blis.NO_TRANSPOSE,
                nr_row,
                nr_col,
                nr_batch,
                one,
                <weight_t*>x,
                nr_row,
                1,
                <weight_t*>y,
                nr_col,
                1,
                one,
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
    elif weights_ft is sparse_weights_t and const_weights_ft is const_sparse_weights_ft:
        for _ in range(nr_batch):
            mm_add_outer(output,
                x, y, nr_row, nr_col)
            x += nr_row
            y += nr_col


