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
    cdef inline void forward(
                        weight_t** fwd_state,
                        const weight_t* weights,
                        const LayerC* layers, 
                        int nr_layer) nogil:
        cdef int i
        cdef LayerC lyr
        for i in range(nr_layer):
            lyr = layers[i]

            lyr.forward(
                fwd_state[i+1],
                &weights[lyr.W],
                fwd_state[i],
                &weights[lyr.bias],
                lyr.nr_out,
                lyr.nr_wide
            )

    @staticmethod
    cdef inline void backward(
                        weight_t** bwd_state,
                        const weight_t** fwd_state,
                        const weight_t* weights,
                        const LayerC* layers,
                        int nr_layer) nogil:
        cdef int i
        cdef LayerC lyr
        # Get layer-wise errors
        for i in range(nr_layer-1, -1, -1):
            lyr = layers[i]
            lyr.backward(
                bwd_state[i],
                bwd_state[i+1],
                fwd_state[i+1],
                &weights[lyr.W],
                lyr.nr_out,
                lyr.nr_wide
            )
        # Need to something like this to back-prop into embeddings
        #MatVec.T_dot(bwd_state[1], weights, bwd_state[0], layers[0].nr_out, layers[0].nr_wide)

    @staticmethod
    cdef inline void set_gradients(
                        weight_t* gradient,
                        const weight_t** bwd_state,
                        const weight_t** fwd_state,
                        const LayerC* layers,
                        int nr_layer) nogil:
        cdef int i
        cdef LayerC lyr
        # Now set the gradients
        for i in range(nr_layer):
            lyr = layers[i]
            MatMat.add_outer_i(&gradient[lyr.W], bwd_state[i], fwd_state[i],
                               lyr.nr_out, lyr.nr_wide)
            VecVec.add_i(&gradient[lyr.bias], bwd_state[i], 1.0, lyr.nr_out)
