from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from .typedefs cimport feat_t, weight_t
from .structs cimport FeatureC, LayerC


cdef class Model:
    cdef PreshMap weights
    cdef Pool mem

    cdef void set_scores(self, weight_t* scores, const FeatureC* feats, int nr_feat) nogil


cdef class LinearModel(Model):
    pass


cdef class MultiLayerPerceptron(Model):
    cdef const LayerC* layers
    cdef int nr_layer
    cdef int nr_all_out
    cdef int nr_all_weight
    cdef int nr_embed

    cdef void backprop(self, weight_t* gradient, weight_t* delta,
        const weight_t* activity, const int* costs) except *

    cdef void set_loss(self, weight_t* delta, const weight_t* activity,
                       const int* costs, int nr_class) nogil

