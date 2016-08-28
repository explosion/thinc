from ..structs cimport LayerC
from ..structs cimport FeatureC
from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t
from ..typedefs cimport weight_t

from ..structs cimport const_weights_ft, const_dense_weights_t, const_sparse_weights_t
from ..structs cimport weights_ft, dense_weights_t, sparse_weights_t


cdef void ReLu_backward(LayerC* gradient, weight_t** bwd,
        const LayerC* weights, const weight_t* const* fwd, const weight_t* randoms,
        const len_t* widths, int nr_layer, int nr_batch, const ConstantsC* hp) nogil
 

cdef void d_affine(weight_t* d_x, weights_ft d_w, weight_t* d_b,
        const weight_t* d_out, const weight_t* x, weights_ft w,
        int nr_out, int nr_in, int nr_batch) nogil
 

cdef void d_softmax(
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

cdef void l2_regularize(weight_t* gradient,
        const weight_t* weights, weight_t strength, int nr_weight) nogil
