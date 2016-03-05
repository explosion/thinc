from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t
from ..typedefs cimport weight_t


cdef void sgd_clip_noise_l2(weight_t* weights, weight_t* moments, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp, weight_t last_update) nogil


cdef void sgd_cm(weight_t* weights, weight_t* moments, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil


cdef void adam(
    weight_t* weights, weight_t* moments, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil

 
cdef void adagrad(
    weight_t* weights, weight_t* moments, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil
 

cdef void adadelta(weight_t* weights, weight_t* momentum, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil
