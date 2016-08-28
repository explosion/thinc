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


