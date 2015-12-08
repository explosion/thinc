from libc.stdint cimport int32_t
from libc.math cimport M_E
from libc.string cimport memset

from .structs cimport LayerC
from .typedefs cimport weight_t
from .blas cimport Vec, VecVec, MatVec, MatMat


cdef class Rectifier:
    @staticmethod
    cdef inline LayerC init(int32_t nr_wide, int32_t nr_out, int32_t offset) nogil:
        cdef LayerC layer
        layer.nr_wide = nr_wide
        layer.nr_out = nr_out
        layer.W = offset
        layer.bias = offset + (nr_wide * nr_out)
        layer.forward = Rectifier.forward
        layer.backward = Rectifier.backward
        return layer

    @staticmethod
    cdef inline void forward(
                        weight_t* output,
                        const weight_t* W,
                        const weight_t* x,
                        const weight_t* bias,
                        int32_t nr_wide,
                        int32_t nr_out) nogil:
        MatVec.dot(output, W, x, nr_wide, nr_out)
        VecVec.add_i(output, bias, 1.0, nr_out)
        cdef int32_t i
        for i in range(nr_out):
            if output[i] < 0:
                output[i] = 0

    @staticmethod
    cdef inline void backward(weight_t* delta_out, weight_t* grad_W, weight_t* grad_b,
                              const weight_t* delta_in,
                              const weight_t* W,
                              const weight_t* fwd_state, 
                              int32_t nr_wide, int32_t nr_out) nogil:

        VecVec.add_i(grad_b, delta_in, 1.0, nr_out)
        MatMat.add_outer_i(grad_W, delta_in, fwd_state, nr_wide, nr_out)

        # Derivative of the rectifier:
        # d_relu(point) = lambda x: 1 if x > 0 else 0
        # 
        # We need to set: 
        #
        # delta = d_relu(signal) * weights_column.dot(delta)
        #
        # Where weights_column are the weights connected to a given output.
        # So, we do the dot product, and then set to 0 where x[i] <= 0
        MatVec.T_dot_i(delta_out, W, nr_wide, nr_out)
        cdef int32_t i
        for i in range(nr_out):
            if fwd_state[i] < 0:
                delta_out[i] = 0


cdef class Softmax:
    @staticmethod
    cdef inline LayerC init(int32_t nr_wide, int32_t nr_out, int32_t offset) nogil:
        cdef LayerC layer
        layer.nr_wide = nr_wide
        layer.nr_out = nr_out
        layer.W = offset
        layer.bias = offset + (nr_wide * nr_out)
        layer.forward = Rectifier.forward
        layer.backward = Rectifier.backward
        return layer

    @staticmethod
    cdef inline void forward(weight_t* output,
                             const weight_t* W,
                             const weight_t* x,
                             const weight_t* bias,
                             int32_t nr_wide,
                             int32_t nr_out) nogil:
        #w = W.dot(actvn) + b
        MatVec.dot(output, W, x, nr_wide, nr_out)
        VecVec.add_i(output, bias, 1.0, nr_out)
        #w = numpy.exp(w - max(w))
        Vec.add_i(output, -Vec.max(output, nr_out), nr_out)
        Vec.exp_i(output, nr_out)
        #w = w / sum(w)
        Vec.div_i(output, Vec.sum(output, nr_out), nr_out)

    @staticmethod
    cdef inline void backward(weight_t* delta_out,
                              weight_t* grad_W,
                              weight_t* grad_b,
                              weight_t* delta_in,
                              const weight_t* W,
                              const weight_t* x, 
                              int32_t nr_wide,
                              int32_t nr_out) nogil:
        MatMat.add_outer_i(grad_W, delta_in, x, nr_wide, nr_out)
        VecVec.add_i(grad_b, delta_in, 1.0, nr_out)
