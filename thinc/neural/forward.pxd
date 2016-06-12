from ..structs cimport FeatureC
from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t
from ..typedefs cimport weight_t


cdef void ELU_forward(weight_t** fwd,
        const weight_t* W, const len_t* shape, int nr_layer, int nr_batch,
        const ConstantsC* hp) nogil
 

cdef void ELU_batch_norm_forward(weight_t** fwd,
        const weight_t* W, const len_t* widths, int nr_layer, int nr_batch,
        const ConstantsC* hp) nogil
 

cdef void ReLu_forward(weight_t** fwd,
        const weight_t* W, const len_t* shape, int nr_below, int nr_above,
        int nr_batch,
        const ConstantsC* hp) nogil
 

cdef void dot_plus(weight_t* out,
        const weight_t* bias, len_t nr_out,
        const weight_t* x, len_t nr_in,
        const weight_t* W) nogil
  

cdef void softmax(weight_t* out, len_t nr_out) nogil
cdef void ELU(weight_t* out, len_t nr_out) nogil
cdef void ReLu(weight_t* out, len_t nr_out) nogil
