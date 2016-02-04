from ..structs cimport FeatureC
from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t


cdef void d_ELU__dot(float* gradient, float** bwd, float* averages,
        const float* W, const float* const* fwd, const len_t* shape,
        int nr_above, int nr_below, const ConstantsC* hp) nogil
   

cdef void d_ReLu__dot(float* gradient, float** bwd, float* averages,
        const float* W, const float* const* fwd, const len_t* shape,
        int nr_above, int nr_below, const ConstantsC* hp) nogil


cdef void d_ELU__dot__normalize__dot(float* gradient, float** bwd, float* averages,
        const float* W, const float* const* fwd, const len_t* shape,
        int nr_above, int nr_below, const ConstantsC* hp) nogil


cdef void d_log_loss(
    float* loss,
        const float* costs,
        const float* scores,
            len_t nr_out) nogil


cdef void d_dot(float* btm_diff,
        int nr_btm,
        const float* top_diff, int nr_top,
        const float* W) nogil
 

cdef void d_ELU(float* delta, const float* signal_out, int n) nogil
