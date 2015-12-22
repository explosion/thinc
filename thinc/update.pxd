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
