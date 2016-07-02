from ..structs cimport FeatureC
from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t
from ..typedefs cimport weight_t


cdef void ELU_backward(weight_t* gradient, weight_t** bwd,
        const weight_t* W, const weight_t* const* fwd, const len_t* shape,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil
   

cdef void ELU_batch_norm_residual_backward(weight_t* G, weight_t** bwd,
        const weight_t* W, const weight_t* const* fwd, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil
 

cdef void ReLu_backward(weight_t* gradient, weight_t** bwd,
        const weight_t* W, const weight_t* const* fwd, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil


cdef void d_log_loss(
    weight_t* loss,
        const weight_t* costs,
        const weight_t* scores,
            len_t nr_out) nogil

cdef void d_hinge_loss(
    weight_t* loss,
        const weight_t* costs,
        const weight_t* scores,
            len_t nr_out) nogil


cdef void d_ELU(weight_t* delta, const weight_t* signal_out, int n) nogil

cdef void d_ReLu(weight_t* delta, const weight_t* signal_out, int n) nogil
