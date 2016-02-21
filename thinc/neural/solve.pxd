from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t
from ..typedefs cimport weight_t


cdef void vanilla_sgd(weight_t* weights, weight_t* moments, weight_t* gradient,
        len_t nr_weight,const ConstantsC* hp) nogil


cdef void backtrack(weight_t* weights, weight_t* diff, weight_t* gradient,
        len_t nr_weight,const ConstantsC* hp) nogil
 
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
