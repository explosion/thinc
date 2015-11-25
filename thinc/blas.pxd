from .typedefs cimport weight_t
from libc.stdint cimport int32_t

cdef class Vec:
    @staticmethod
    cdef inline void mul(weight_t* output, const weight_t* vec, weight_t scal,
                         int32_t nr) nogil:
        pass

    @staticmethod
    cdef inline void mul_i(weight_t* vec, const weight_t scal, int32_t nr) nogil:
        pass


cdef class VecVec:
    @staticmethod
    cdef inline void add(weight_t* output,
                         const weight_t* x, 
                         const weight_t* y,
                         int32_t nr) nogil:
        pass
   
    @staticmethod
    cdef inline void add_i(weight_t* x, 
                           const weight_t* y,
                           int32_t nr) nogil:
        pass
 
    @staticmethod
    cdef inline void add_pow(weight_t* output,
                         const weight_t* x, 
                         const weight_t* y,
                         weight_t power,
                         int32_t nr) nogil:
        pass
   
    @staticmethod
    cdef inline void add_pow_i(weight_t* x, 
                               const weight_t* y,
                               weight_t power,
                               int32_t nr) nogil:
        pass
 
    @staticmethod
    cdef inline void mul(weight_t* output,
                         const weight_t* x, 
                         const weight_t* y,
                         int32_t nr_row) nogil:
        pass
   
    @staticmethod
    cdef inline void mul_i(weight_t* x, 
                           const weight_t* y,
                           int32_t nr_row) nogil:
        pass
 
    @staticmethod
    cdef inline weight_t dot(const weight_t* x, 
                             const weight_t* y,
                             int32_t nr_row) nogil:
        pass
 

cdef class MatVec:
    @staticmethod
    cdef inline void mul(weight_t* output,
                         const weight_t* mat,
                         const weight_t* vec,
                         int32_t nr_row, int32_t nr_col) nogil:
        pass

    @staticmethod
    cdef inline void mul_i(weight_t* mat,
                           const weight_t* vec,
                           int32_t nr_row, int32_t nr_col) nogil:
        pass

    @staticmethod
    cdef inline void dot(weight_t* output,
                         const weight_t* mat,
                         const weight_t* vec,
                         int32_t nr_row, int32_t nr_col) nogil:
        pass


cdef class MatMat:
    @staticmethod
    cdef inline void add(weight_t* output,
                         const weight_t* x,
                         const weight_t* y,
                         int32_t nr_row, int32_t nr_col) nogil:
        pass

    @staticmethod
    cdef inline void add_i(weight_t* x,
                           const weight_t* y,
                           int32_t nr_row, int32_t nr_col) nogil:
        pass

    @staticmethod
    cdef inline void mul(weight_t* output,
                         const weight_t* x,
                         const weight_t* y,
                         int32_t nr_row, int32_t nr_col) nogil:
        pass

    @staticmethod
    cdef inline void mul_i(weight_t* x,
                           const weight_t* y,
                           int32_t nr_row, int32_t nr_col) nogil:
        pass

    @staticmethod 
    cdef inline void add_outer() nogil:
        pass

    @staticmethod 
    cdef inline void add_outer_i() nogil:
        pass
