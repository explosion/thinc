from cymem.cymem cimport Pool

from preshed.maps cimport PreshMap
from .typedefs cimport time_t, feat_t, weight_t, class_t
from .api cimport ExampleC
#from .structs cimport OptimizerC, MapC


cdef class Updater:
    cdef public int time
    cdef Pool mem
    cdef PreshMap train_weights
    cdef PreshMap weights
    
    cdef void update(self, ExampleC* eg) except *

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1


cdef class AveragedPerceptronUpdater(Updater):
    pass


#cdef class Optimizer:
#    cdef Pool mem
#    cdef OptimizerC c
#
#    cdef void rescale(self, weight_t* gradient, weight_t* support, int nr_weight) nogil
#
#    cdef void update(self, weight_t* weights, weight_t* gradient, weight_t* support,
#                     int nr_weight) nogil
#
#    cdef void update_sparse(self, MapC* weights, MapC* gradients, MapC* supports,
#                            int length) nogil
#
#
#cdef class Adagrad(Optimizer):
#    pass
