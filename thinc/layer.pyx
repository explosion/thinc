from .typedefs cimport weight_t

cimport numpy as np
import numpy


#def relu(weight_t[:] weights, nr_out, nr_wide, weight_t[:] signal):
#    cdef LayerC layer = Rectifier.init(nr_out, nr_wide, 0)
#    
#    cdef np.ndarray output = numpy.zeros(shape=(nr_out,), dtype='float32')
#
#    Rectifier.forward(
#        <weight_t*>output.data,
#        &weights[layer.W],
#        &signal[0],
#        &weights[layer.bias],
#        layer.nr_out,
#        layer.nr_wide
#    )
#    return output
#
#
#def d_relu(weight_t[:] delta_in, weight_t[:] signal_out, weight_t[:] W,
#           int32_t nr_out, int32_t nr_wide):
#        
#    cdef LayerC layer = Rectifier.init(nr_out, nr_wide, 0)
#    
#    cdef np.ndarray output = numpy.zeros(shape=(nr_wide,), dtype='float32')
#
#    Rectifier.backward(
#        <weight_t*>output.data,
#        &delta_in[0],
#        &signal_out[0],
#        &W[0],
#        layer.nr_out,
#        layer.nr_wide
#    )
#    return output
#
#
#def softmax(weight_t[:] weights, nr_out, nr_wide, weight_t[:] signal):
#    cdef LayerC layer = Softmax.init(nr_out, nr_wide, 0)
#    
#    cdef np.ndarray output = numpy.zeros(shape=(nr_out,), dtype='float32')
#
#    Softmax.forward(
#        <weight_t*>output.data,
#        &weights[layer.W],
#        &signal[0],
#        &weights[layer.bias],
#        layer.nr_out,
#        layer.nr_wide
#    )
#    return output
#
#
#def d_softmax(weight_t[:] delta_in, weight_t[:] signal_out, weight_t[:] W,
#           int32_t nr_out, int32_t nr_wide):
#        
#    cdef LayerC layer = Rectifier.init(nr_out, nr_wide, 0)
#    
#    cdef np.ndarray output = numpy.zeros(shape=(nr_out,), dtype='float32')
#
#    Softmax.backward(
#        <weight_t*>output.data,
#        &delta_in[0],
#        &signal_out[0],
#        &W[0],
#        layer.nr_out,
#        layer.nr_wide
#    )
#    return output
