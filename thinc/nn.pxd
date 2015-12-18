cimport cython
from libc.string cimport memset, memcpy
from libc.math cimport sqrt as c_sqrt
from libc.stdint cimport int32_t

from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from .api cimport Learner
from .structs cimport NeuralNetC, ExampleC, FeatureC, LayerC, HyperParamsC
from .typedefs cimport weight_t, atom_t
from .api cimport Example
from .blas cimport MatMat, MatVec, VecVec


cdef class NeuralNet(Learner):
    cdef NeuralNetC c
    
    cdef Pool mem
    cdef PreshMap weights
    cdef PreshMap train_weights

    @staticmethod
    cdef inline void forward(weight_t** fwd, const LayerC* layers, int nr) nogil:
        cdef int i
        cdef LayerC lyr
        for i in range(nr):
            lyr = layers[i]
            lyr.forward(
                fwd[i+1], # Len=nr_out
                fwd[i], # Len=nr_wide
                lyr.W, # Len=nr_out * nr_wide
                lyr.b, # Len=nr_out
                lyr.nr_out,
                lyr.nr_wide)

    @staticmethod
    cdef inline void backward(weight_t** bwd, const weight_t** fwd, 
                              const LayerC* layers, int nr) nogil:
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
        cdef int i
        cdef LayerC lyr
        # Get layer-wise errors
        for i in range(nr-1, 0, -1):
            lyr = layers[i]
            lyr.backward(
                bwd[i],     # Output: error of this layer, len=width
                bwd[i+1],   # Input: error from layer above, len=nr_out
                fwd[i],     # Input: signal from layer below, len=nr_wide
                lyr.W,      # Weights of this layer
                lyr.nr_out, # Width of next layer 
                lyr.nr_wide # Width of prev layer 
            )

    @staticmethod
    cdef inline void set_gradients(weight_t* gradient, const weight_t** bwd,
                                   const weight_t** fwd, const LayerC* layers,
                                   int nr_layer) nogil:
        cdef int i
        cdef LayerC lyr
        # Now set the gradients
        for i in range(nr_layer):
            lyr = layers[i]
            MatMat.add_outer_i(gradient, bwd[i+1], fwd[i], lyr.nr_out, lyr.nr_wide)
            gradient += lyr.nr_out * lyr.nr_wide
            VecVec.add_i(gradient, bwd[i+1], 1.0, lyr.nr_out)
            gradient += lyr.nr_out
