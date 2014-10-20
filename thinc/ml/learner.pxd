from libc.stdint cimport uint64_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uint16_t

from cymem.cymem cimport Pool

from preshed.maps cimport PreshMap
from preshed.maps cimport PreshMapArray
from preshed.maps cimport MapStruct
from preshed.maps cimport Cell

from thinc.cache cimport ScoresCache

from .weights cimport WeightLine
from .weights cimport TrainFeat


ctypedef int weight_t
ctypedef uint64_t feat_t
ctypedef uint32_t class_t
ctypedef uint32_t count_t
ctypedef uint32_t time_t


cdef class LinearModel:
    cdef time_t time
    cdef readonly class_t nr_class
    cdef size_t n_corr
    cdef size_t total
    cdef Pool mem
    cdef PreshMapArray weights
    cdef ScoresCache cache
    cdef weight_t* scores
    cdef WeightLine** _weight_lines

    cdef TrainFeat* new_feat(self, size_t template_id, feat_t feat_id) except NULL
    cdef class_t score(self, weight_t* scores, feat_t* features, weight_t* values,
            int n_feats) except *
    cpdef int update(self, dict counts) except -1
