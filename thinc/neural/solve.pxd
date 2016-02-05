from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t


cdef void vanilla_sgd(float* weights, float* moments, float* gradient,
        len_t nr_weight,const ConstantsC* hp) nogil


cdef void sgd_cm(float* weights, float* moments, float* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil


cdef void adam(
    float* weights, float* moments, float* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil

 
cdef void adagrad(
    float* weights, float* moments, float* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil
 

cdef void adadelta(float* weights, float* momentum, float* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil
