cimport cython
from libc.string cimport memset, memcpy
from libc.math cimport sqrt as c_sqrt

from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from .api cimport Learner
from .structs cimport ExampleC, FeatureC, LayerC, HyperParamsC
from .typedefs cimport weight_t, atom_t
from .api cimport Example
from .blas cimport MatMat, MatVec, VecVec


cdef class NeuralNetwork(Learner):
    cdef Pool mem
    cdef PreshMap weights
    cdef PreshMap train_weights
    cdef LayerC* layers
    cdef HyperParamsC hyper_params
    cdef int nr_dense
    cdef int nr_layer
 
    
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
                fwd_state[i],
                &weights[lyr.W],
                &weights[lyr.bias],
                lyr.nr_wide,
                lyr.nr_out
            )

    @staticmethod
    cdef inline void set_loss(weight_t* loss, const weight_t* scores, int best,
                              int nr_class) nogil:
        # Here we'll take a little short-cut, and for now say the loss is the
        # weight assigned to the 'best'  class
        # Probably we want to give credit for assigning weight to other correct
        # classes
        cdef int i
        for i in range(nr_class):
            loss[i] = (i == best) - scores[i]

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
                fwd_state[i],
                &weights[lyr.W],
                lyr.nr_out,
                lyr.nr_wide
            )

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
