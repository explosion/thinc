from libc.stdint cimport int32_t
from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap
from .typedefs cimport time_t, feat_t, weight_t, class_t
from .api cimport ExampleC


cdef class Updater:
    cdef public int time
    cdef Pool mem
    cdef PreshMap train_weights
    cdef PreshMap weights
    
    cdef void update(self, ExampleC* eg) except *

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1


cdef class AveragedPerceptronUpdater(Updater):
    pass


cdef class DenseUpdater(Updater):
    cdef readonly int nr_dense
    cdef public weight_t eta
    cdef public weight_t eps
    cdef public weight_t rho
    
    cdef void _update(self, weight_t* weights, void* support, weight_t* gradient,
            int32_t n) except *


cdef class Adagrad(DenseUpdater):
    pass
