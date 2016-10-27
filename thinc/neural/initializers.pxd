from ..typedefs cimport weight_t

cdef void he_normal_initializer(weight_t* weights, int fan_in, int n) except *

cdef void he_uniform_initializer(weight_t* weights,
    weight_t low, weight_t high, int n) except *

cdef void constant_initializer(weight_t* weights, weight_t value, int n) nogil
