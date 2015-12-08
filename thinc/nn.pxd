cimport cython
from libc.stdint cimport int32_t
from libc.string cimport memset, memcpy
from libc.math cimport sqrt as c_sqrt

from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from .api cimport Learner
from .structs cimport ExampleC, FeatureC, LayerC, HyperParamsC
from .typedefs cimport weight_t, atom_t
from .api cimport Example


cdef class NeuralNetwork(Learner):
    cdef Pool mem
    cdef PreshMap weights
    cdef PreshMap train_weights
    cdef LayerC* layers
    cdef HyperParamsC hyper_params
    cdef int32_t nr_dense
    cdef int32_t nr_layer
 
