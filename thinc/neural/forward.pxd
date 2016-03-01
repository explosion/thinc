from ..structs cimport FeatureC
from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t


cdef void dot_plus__ELU(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_below, int nr_above,
        const ConstantsC* hp) nogil
 

cdef void dot_plus__ReLu(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_below, int nr_above,
        const ConstantsC* hp) nogil
 

cdef void dot_plus__residual__ELU(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_below, int nr_above,
        const ConstantsC* hp) nogil


cdef void dot__normalize__dot_plus__ELU(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_before, int nr_above,
        const ConstantsC* hp) nogil


cdef void dot_plus(float* out,
        const float* bias, len_t nr_out,
        const float* x, len_t nr_in,
        const float* W) nogil
  

cdef void softmax(float* out, len_t nr_out) nogil


cdef void ELU(float* out, len_t nr_out) nogil
