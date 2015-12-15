from libc.stdint cimport int32_t
from libc.math cimport M_E
from libc.string cimport memset

from preshed.maps cimport MapStruct, map_get

from .structs cimport LayerC, FeatureC
from .typedefs cimport weight_t
from .blas cimport Vec, VecVec, MatVec, MatMat
from .api cimport arg_max_if_zero


cdef class Embedding:
    @staticmethod
    cdef inline void set_layer(weight_t* output, const MapStruct* map_,
                               const FeatureC* feats, int32_t nr_feat) nogil:
        cdef int32_t i, j
        for i in range(nr_feat):
            feat = feats[i]
            feat_embed = <const weight_t*>map_get(map_, feat.key)
            if feat_embed is not NULL:
                VecVec.add_i(&output[feat.i], feat_embed, feat.val, feat.length)
            else:
                for j in range(feat.length):
                    output[feat.i + j] = 0.25


cdef class Rectifier:
    @staticmethod
    cdef inline LayerC init(int32_t nr_out, int32_t nr_wide, int32_t offset) nogil:
        cdef LayerC layer
        layer.nr_out = nr_out
        layer.nr_wide = nr_wide
        layer.W = offset
        layer.bias = offset + (nr_wide * nr_out)
        layer.forward = Rectifier.forward
        layer.backward = Rectifier.backward
        return layer

    @staticmethod
    cdef inline void forward(
                        weight_t* out,
                        const weight_t* W,
                        const weight_t* in_,
                        const weight_t* bias,
                        int32_t nr_out,
                        int32_t nr_wide) nogil:
        # We're a layer of M cells, which we can think of like classes
        # Each class sums over N inputs, which we can think of as features
        # Each feature has a weight. So we own M*N weights
        # We receive an input vector of N dimensions. We produce an output vector
        # of M activations.
        MatVec.dot(out, W, in_, nr_out, nr_wide)
        VecVec.add_i(out, bias, 1.0, nr_out)
        cdef int32_t i
        for i in range(nr_out):
            # Writing this way handles NaN
            if not (out[i] > 0):
                out[i] = 0

    @staticmethod
    cdef inline void backward(
                        weight_t* delta_out,
                        const weight_t* delta_in,
                        const weight_t* signal_out,
                        const weight_t* W,
                        int32_t nr_out,
                        int32_t nr_wide) nogil:
        # delta = W.T.dot(prev_delta) * d_relu(signal_in)
        # d_relu(signal_in) is a binary vector, 0 when signal_in < 0
        # So, we do our dot product, and then clip to 0 on the dimensions where
        # signal_in is 0
        # Note that prev_delta is a column vector (the error of our output),
        # while delta is a row vector (the error of our neurons, which must match
        # the input layer's width)
        MatVec.T_dot(delta_out, W, delta_in, nr_out, nr_wide)
        cdef int32_t i
        for i in range(nr_wide):
            if signal_out[i] < 0:
                delta_out[i] = 0


cdef class Softmax:
    @staticmethod
    cdef inline LayerC init(int32_t nr_out, int32_t nr_wide, int32_t offset) nogil:
        cdef LayerC layer
        layer.nr_out = nr_out
        layer.nr_wide = nr_wide
        layer.W = offset
        layer.bias = offset + (nr_out * nr_wide)
        layer.forward = Softmax.forward
        layer.backward = Softmax.backward
        return layer

    @staticmethod
    cdef inline void forward(weight_t* out,
                             const weight_t* W,
                             const weight_t* in_,
                             const weight_t* bias,
                             int32_t nr_out,
                             int32_t nr_wide) nogil:
        #w = W.dot(actvn) + b
        MatVec.dot(out, W, in_, nr_out, nr_wide)
        VecVec.add_i(out, bias, 1.0, nr_out)
        #w = numpy.exp(w - max(w))
        Vec.add_i(out, -Vec.max(out, nr_out), nr_out)
        Vec.exp_i(out, nr_out)
        #w = w / sum(w)
        Vec.div_i(out, Vec.sum(out, nr_out), nr_out)

    @staticmethod
    cdef inline void backward(
                        weight_t* loss,
                        const weight_t* costs,
                        const weight_t* scores,
                        const weight_t* __W_unused,
                        int32_t nr_out,
                        int32_t nr_wide) nogil:
        '''Compute derivative of log loss'''
        # Here we'll take a little short-cut, and for now say the loss is the
        # weight assigned to the 'best'  class
        # Probably we want to give credit for assigning weight to other correct
        # classes
        cdef int i
        for i in range(nr_out):
            loss[i] = -scores[i]
        cdef int best = arg_max_if_zero(scores, costs, nr_out)
        # We could branch in the loop, but this is probably faster
        loss[best] = 1.0 - scores[best]
