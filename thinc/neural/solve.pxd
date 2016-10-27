from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..structs cimport SparseArrayC
from ..structs cimport const_weights_ft, const_dense_weights_t, const_sparse_weights_t
from ..structs cimport weights_ft, dense_weights_t, sparse_weights_t

cdef void sgd_cm(weights_ft weights, weights_ft gradient,
        len_t nr_weight, const ConstantsC* hp) nogil


cdef void nag(weights_ft weights, weights_ft gradient,
        len_t nr_weight, const ConstantsC* hp) nogil


cdef void adam(weights_ft weights, weights_ft gradient,
        len_t nr_weight, const ConstantsC* hp) nogil
