from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from .features cimport ConjunctionExtracter
from ..typedefs cimport weight_t, feat_t, class_t
from ..structs cimport FeatureC
from ..structs cimport ExampleC


cdef class AveragedPerceptron:
    cdef readonly Pool mem
    cdef readonly PreshMap weights
    cdef readonly PreshMap averages
    cdef readonly PreshMap lasso_ledger
    cdef ConjunctionExtracter extracter
    cdef public int time
    cdef public weight_t learn_rate
    cdef public weight_t l1_penalty
    cdef public weight_t momentum
    
    cdef void set_scoresC(self, weight_t* scores, const FeatureC* feats, int nr_feat) nogil
    cdef int updateC(self, const ExampleC* eg) except -1
    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1
