from ..typedefs cimport weight_t, len_t


cdef void ELU(weight_t* out, len_t nr_out) nogil
cdef void ReLu(weight_t* out, len_t nr_out) nogil
cdef void softmax(weight_t* out, len_t nr_out) nogil
cdef void d_ReLu(weight_t* delta, const weight_t* signal_out, int n) nogil
