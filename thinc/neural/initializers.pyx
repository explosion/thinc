from ..typedefs cimport weight_t
import numpy


cdef void he_normal_initializer(weight_t* weights, int fan_in, int n) except *:
    # See equation 10 here:
    # http://arxiv.org/pdf/1502.01852v1.pdf
    values = numpy.random.normal(loc=0.0, scale=numpy.sqrt(2.0 / float(fan_in)), size=n)
    cdef weight_t value
    for i, value in enumerate(values):
        weights[i] = value


cdef void he_uniform_initializer(weight_t* weights,
        weight_t low, weight_t high, int n) except *:
    values = numpy.random.uniform(size=n, low=low, high=high)
    cdef weight_t value
    for i, value in enumerate(values):
        weights[i] = value


cdef void constant_initializer(weight_t* weights, weight_t value, int n) nogil:
    for i in range(n):
        weights[i] = value
