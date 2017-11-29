cdef struct params_s:
    float* weights
    float* gradient
    params_s* prev
    params_s* next
    int nr_grad_upd
    int nr_weight_upd
    int ref_count
    int N


cdef params_s init_params(int N) nogil

cdef void free_params(params_s* params) nogil

cdef params_s* refresh_params(params_s* params) nogil
