# cython: infer_types=True
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy, memset


cdef struct params_s:
    float* weights
    float* gradient
    params_s* prev
    params_s* next
    int nr_grad_upd
    int nr_weight_upd
    int ref_count
    int N


cdef params_s init_params(int N) nogil:
    cdef params_s params
    params.weights = <float*>calloc(N, sizeof(float))
    params.gradient = <float*>calloc(N, sizeof(float))
    params.prev = NULL
    params.next = NULL
    params.nr_grad_upd = 0
    params.nr_weight_upd = 0
    params.ref_count = 1
    params.N = N
    return params

cdef void free_params(params_s* params) nogil:
    params.next.prev = NULL
    free(params.weights)
    free(params.gradient)
    free(params)

cdef params_s* refresh_params(params_s* params) nogil:
    params.next.ref_count += 1
    params = params.next
    if params.prev.ref_count > 1:
        params.prev.ref_count -= 1
    else:
        # If we're the last reference, try to free unreferred chains. We can't
        # free a link if one of its backpointers has a ref count >= 1
        ptr = params.prev
        while ptr.prev != NULL:
            # We can't free the chain.
            if ptr.ref_count >= 1:
                break
            else:
                ptr = ptr.prev
        else:
            # We can free the chain -- so walk forward, freeing it.
            ptr = ptr.next
            while ptr != params:
                ptr = ptr.next
                free_params(ptr.prev)
    return params


