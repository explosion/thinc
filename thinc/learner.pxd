from libc.stdio cimport FILE

from cymem.cymem cimport Pool

from preshed.maps cimport PreshMapArray
from preshed.maps cimport MapStruct
from preshed.maps cimport Cell

from .cache cimport ScoresCache

from .weights cimport WeightLine
from .weights cimport TrainFeat
from .typedefs cimport *
from .features cimport Feature


DEF LINE_SIZE = 8


cdef class LinearModel:
    cdef time_t time
    cdef readonly class_t nr_class
    cdef readonly int nr_templates
    cdef size_t n_corr
    cdef size_t total
    cdef Pool mem
    cdef PreshMapArray weights
    cdef ScoresCache cache
    cdef weight_t* scores
    cdef WeightLine* _weight_lines
    cdef size_t _max_wl

    cdef int set_scores(self, weight_t* scores, Feature* feats, int n_feats) except -1
    cdef weight_t* get_scores(self, Feature* feats, int n_feats) except NULL
    cpdef int update(self, dict counts) except -1


cdef class _Writer:
    cdef FILE* _fp
    cdef class_t _nr_class
    cdef count_t _freq_thresh

    cdef int write(self, int i, feat_t feat_id, TrainFeat* feat) except -1


cdef class _Reader:
    cdef FILE* _fp
    cdef class_t _nr_class
    cdef count_t _freq_thresh

    cdef int read(self, Pool mem, int* out_i, feat_t* out_id, TrainFeat** out_feat) except -1
