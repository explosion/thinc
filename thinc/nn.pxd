cimport cython
from libc.string cimport memset, memcpy
from libc.math cimport sqrt as c_sqrt
from libc.stdint cimport int32_t

from cymem.cymem cimport Pool

from .structs cimport NeuralNetC
from .typedefs cimport weight_t
from .blas cimport Vec, MatMat, MatVec, VecVec

# The input/output of the fwd/bwd pass can be confusing. Some notes.
#
# Forward pass. in0 is at fwd_state[0]. Activation of layer 1 is
# at fwd_state[1]
# 
# in0 = input_
# in1 = act0 = ReLu(in0 * W0 + b0)
# in2 = act1 = ReLu(in1 * W1 + b1)
# out = act2 = Softmax(in2 * W2 + b2)

# Okay so our scores are at fwd_state[3]. Our loss will live there too.
# The loss will then be used to calculate the gradient for layer 2.
# We now sweep backward, and calculate the next loss, which will be used
# to calculate the gradient for layer 1, etc.
#
# So, the total loss is at bwd_state[3]
# 
# g2 = d3 = out - target
# g1 = d2 = Back(d3, in2, w2, b2)
# g0 = d1 = Back(d2, in1, w1, b1)
# gE = d0 = Back(d1, in0, w0, b0)
# 
# gE here refers to the 'fine tuning' vector, for word embeddings



cdef class NeuralNet:
    cdef Pool mem
    
    cdef NeuralNetC c

    @staticmethod
    cdef inline void forward_backward(
            weight_t* gradient, weight_t** fwd_acts, weight_t** bwd_acts,
            const weight_t* input_, const weight_t* costs, const NeuralNetC* nn) nogil:
        # Ensure the fwd_state and bwd_state buffers are wiped
        for i in range(nn.nr_layer):
            memset(fwd_acts[i], 0, nn.widths[i] * sizeof(weight_t))
            memset(bwd_acts[i], 0, nn.widths[i] * sizeof(weight_t))
        for i in range(nn.widths[0]):
            fwd_acts[0][i] = input_[i]

        NeuralNet.forward(fwd_acts,
            nn.weights, nn.widths, nn.nr_layer)

        Softmax.delta_log_loss(bwd_acts[nn.nr_layer-1],
            costs, fwd_acts[nn.nr_layer-1], nn.widths[nn.nr_layer-1])
        
        NeuralNet.backward(bwd_acts,
            fwd_acts, nn.weights + nn.nr_weight, nn.widths, nn.nr_layer)
        
        NeuralNet.set_gradient(gradient,
            bwd_acts, fwd_acts, nn.widths, nn.nr_layer)
  

    @staticmethod
    cdef inline void forward(weight_t** fwd,
                        const weight_t* W,
                        const int* widths, int n) nogil:
        cdef int i
        for i in range(n-2): # Save last layer for softmax
            Rectifier.forward(fwd[i+1], fwd[i], W, widths[i+1], widths[i])
            W += widths[i+1] * widths[i] + widths[i+1]
        Softmax.forward(fwd[n-1], fwd[n-2], W, widths[n-1], widths[n-2])

    @staticmethod
    cdef inline void backward(weight_t** bwd,
                        const weight_t* const* fwd, 
                        const weight_t* W,
                        const int* widths, int n) nogil:
        cdef int i
        for i in range(n-2, -1, -1):
            W -= widths[i+1] * widths[i] + widths[i+1]
            Rectifier.backward(bwd[i], # Output: error of this layer, len=width
                bwd[i+1],    # Input: error from layer above, len=nr_out
                fwd[i],      # Input: signal from layer below, len=nr_wide
                W,           # Weights of this layer
                widths[i+1], # Width of next layer 
                widths[i]    # Width of this layer 
            )

    @staticmethod
    cdef inline void set_gradient(weight_t* gradient,
                        const weight_t* const* bwd,
                        const weight_t* const* fwd,
                        const int* widths, int n) nogil:
        cdef int i
        # Now set the gradients
        for i in range(n-1):
            MatMat.add_outer_i(gradient, bwd[i+1], fwd[i], widths[i+1], widths[i])
            gradient += widths[i+1] * widths[i]
            VecVec.add_i(gradient, bwd[i+1], 1.0, widths[i+1])
            gradient += widths[i+1]


cdef class Rectifier:
    @staticmethod
    cdef inline void forward(weight_t* out,
                        const weight_t* in_,
                        const weight_t* W,
                        int32_t nr_out,
                        int32_t nr_wide) nogil:
        # We're a layer of M cells, which we can think of like classes
        # Each class sums over N inputs, which we can think of as features
        # Each feature has a weight. So we own M*N weights
        # We receive an input vector of N dimensions. We produce an output vector
        # of M activations.
        MatVec.dot(out, W, in_, nr_out, nr_wide)
        # Bias
        VecVec.add_i(out, W + (nr_out * nr_wide), 1.0, nr_out)
        cdef int32_t i
        for i in range(nr_out):
            # Writing this way handles NaN
            if not (out[i] > 0):
                out[i] = 0

    @staticmethod
    cdef inline void backward(weight_t* delta_out,       # Len == nr_wide
                        const weight_t* delta_in,  # Len == nr_out
                        const weight_t* signal_in, # Len == nr_wide
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
            if signal_in[i] < 0:
                delta_out[i] = 0


cdef class Softmax:
    @staticmethod
    cdef inline void forward(weight_t* out,
                             const weight_t* in_,
                             const weight_t* W,
                             int32_t nr_out,
                             int32_t nr_wide) nogil:
        #w = W.dot(actvn) + b
        MatVec.dot(out, W, in_, nr_out, nr_wide)
        # Bias
        VecVec.add_i(out, W + (nr_out * nr_wide), 1.0, nr_out)
        #w = numpy.exp(w - max(w))
        Vec.add_i(out, -Vec.max(out, nr_out), nr_out)
        Vec.exp_i(out, nr_out)
        #w = w / sum(w)
        Vec.div_i(out, Vec.sum(out, nr_out), nr_out)

    @staticmethod
    cdef inline void delta_log_loss(weight_t* loss,
                        const weight_t* costs,
                        const weight_t* scores,
                        int32_t nr_out) nogil:
        '''Compute derivative of log loss'''
        # Here we'll take a little short-cut, and for now say the loss is the
        # weight assigned to the 'best'  class
        # Probably we want to give credit for assigning weight to other correct
        # classes
        cdef int i
        for i in range(nr_out):
            loss[i] = scores[i] - (costs[i] == 0)


cdef class Adagrad:
    @staticmethod
    @cython.cdivision(True)
    cdef inline void update(weight_t* weights, weight_t* gradient, weight_t* support,
                            int32_t n, weight_t eta, weight_t eps) nogil:
        '''
        Update weights with Adagrad
        '''
        VecVec.add_pow_i(support, gradient, 2.0, n)
        cdef int i
        for i in range(n):
            gradient[i] *= eta / (c_sqrt(support[i]) + eps)
        VecVec.add_i(weights, gradient, -1.0, n)
