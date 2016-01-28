from ..structs cimport FeatureC
from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t


cdef extern from "math.h" nogil:
    float expf(float x)
    float sqrtf(float x)


cdef void dot_plus__ELU(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_below, int nr_above,
        const ConstantsC* hp) nogil
 

cdef void dot_plus__ReLu(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_below, int nr_above,
        const ConstantsC* hp) nogil
 

cdef void dot_plus__residual__ELU(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_below, int nr_above,
        const ConstantsC* hp) nogil


cdef void d_ELU__dot(float* gradient, float** bwd, float* averages,
        const float* W, const float* const* fwd, const len_t* shape,
        int nr_above, int nr_below, const ConstantsC* hp) nogil
   

cdef void d_ReLu__dot(float* gradient, float** bwd, float* averages,
        const float* W, const float* const* fwd, const len_t* shape,
        int nr_above, int nr_below, const ConstantsC* hp) nogil


cdef void dot__normalize__dot_plus__ELU(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_before, int nr_above,
        const ConstantsC* hp) nogil


cdef void d_ELU__dot__normalize__dot(float* gradient, float** bwd, float* averages,
        const float* W, const float* const* fwd, const len_t* shape,
        int nr_above, int nr_below, const ConstantsC* hp) nogil
 


cdef void dot_plus(float* out,
        const float* bias, len_t nr_out,
        const float* x, len_t nr_in,
        const float* W) nogil


cdef void d_dot(float* btm_diff,
        int nr_btm,
        const float* top_diff, int nr_top,
        const float* W) nogil


cdef void ELU(float* out, len_t nr_out) nogil


cdef void d_ELU(float* delta, const float* signal_out, int n) nogil


cdef void ReLu(float* out, len_t nr_out) nogil


cdef void d_ReLu(float* delta, const float* signal_out, int n) nogil


cdef void softmax(float* out, len_t nr_out) nogil


cdef void d_log_loss(
    float* loss,
        const float* costs,
        const float* scores,
            len_t nr_out) nogil


cdef void adam(
    float* weights, float* moments, float* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil

 
cdef void adagrad(
    float* weights, float* moments, float* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil
 

cdef void adadelta(float* weights, float* momentum, float* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil
 

cdef void vanilla_sgd_update_step(float* weights, float* moments, float* gradient,
        len_t nr_weight,const ConstantsC* hp) nogil


cdef void normalize(float* x_norm, float* Ex, float* Vx,
        const float* x, len_t nr_x, float alpha, float time) nogil


cdef void d_normalize(float* bwd, float* E_dEdXh, float* E_dEdXh_dot_Xh,
        const float* Xh, const float* Vx, len_t n, float alpha, float time) nogil


