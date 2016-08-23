from libc.stdint cimport uint64_t

from ..base cimport Model
from ..typedefs cimport weight_t, feat_t, class_t
from ..structs cimport NeuralNetC, FeatureC, MinibatchC, ExampleC
from ..extra.eg cimport Example
from ..extra.mb cimport Minibatch

from cymem.cymem cimport Pool


cdef class NeuralNet(Model):
    cdef readonly Pool mem
    cdef readonly Minibatch _mb
    cdef NeuralNetC c

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1
    
    cdef void set_scoresC(self, weight_t* scores,
        const void* feats, int nr_feat, int is_sparse) nogil

    cdef weight_t updateC(self, const void* feats, int nr_feat, int is_sparse,
        const weight_t* costs, const int* is_valid, int force_update, uint64_t key) except -1

    cdef void _updateC(self, MinibatchC* mb) except *

    cdef void _extractC(self, weight_t* input_,
        const void* feats, int nr_feat, int is_sparse) nogil
    
    cdef void _softmaxC(self, weight_t* output) nogil
    
    cdef void _set_delta_lossC(self, weight_t* delta_loss,
        const weight_t* costs, const weight_t* scores) nogil

    cdef void _backprop_extracterC(self, const weight_t* deltas, const void* feats,
            int nr_feat, int is_sparse) nogil
 
    cdef void _update_extracterC(self, const void* feats,
            int nr_feat, int batch_size, int is_sparse) nogil
