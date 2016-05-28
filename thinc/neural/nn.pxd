from ..base cimport Model
from ..typedefs cimport weight_t, feat_t, class_t
from ..structs cimport NeuralNetC, FeatureC, MinibatchC, ExampleC
from ..extra.eg cimport Example
from ..extra.mb cimport Minibatch

from cymem.cymem cimport Pool


cdef class NeuralNet(Model):
    cdef readonly Pool mem
    cdef NeuralNetC c
    cdef MinibatchC* _mb

    cdef void set_scoresC(self, weight_t* scores,
        const void* feats, int nr_feat, int is_sparse) nogil

    cdef Minibatch updateC(self, const void* feats, int nr_feat, int is_sparse,
        weight_t* costs, int* is_valid, int force_update)

    cdef int _updateC(self, MinibatchC* mb) except -1

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1
