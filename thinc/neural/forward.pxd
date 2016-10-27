from libc.stdint cimport uint64_t
from ..structs cimport LayerC
from ..structs cimport FeatureC
from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..structs cimport LayerC, SparseArrayC
from ..structs cimport const_weights_ft, const_dense_weights_t, const_sparse_weights_t
from ..structs cimport weights_ft, dense_weights_t, sparse_weights_t


cdef void ReLu_forward(weight_t** fwd,
        const LayerC* weights, const weight_t* randoms, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil


#cdef void ELU_batch_norm_residual_forward(weight_t** fwd,
#        const weight_t* W, const weight_t* randoms, const len_t* widths,
#        int nr_layer, int nr_batch, const ConstantsC* hp) nogil
# 

#cdef void ReLu_layer_norm_forward(weight_t** fwd,
#        const weight_t* W, const weight_t* randoms, const len_t* widths,
#        int nr_layer, int nr_batch, const ConstantsC* hp) nogil
# 
#

#cdef int skip_layer(weight_t timestep, uint64_t layer, int nr_in, int nr_out) nogil
#
#cdef void normalize(weight_t* x, const weight_t* Ex, const weight_t* Vx,
#        int nr_out, int nr_batch) nogil
#
#cdef void layer_normalize(weight_t* x, int nr_out, int nr_batch) nogil
#



cdef void softmax(weight_t* out, len_t nr_out, len_t nr_batch) nogil

cdef void ELU(weight_t* out, len_t nr_out, len_t nr_batch) nogil
cdef void ReLu(weight_t* out, len_t nr_out, len_t nr_batch) nogil

cdef void affine(weight_t* out,
        const weight_t* in_, const_weights_ft W, const weight_t* bias,
        int nr_out, int nr_in, int nr_batch) nogil
