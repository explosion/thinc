from libc.stdio cimport FILE
from libc.string cimport memset

from cymem.cymem cimport Pool

from preshed.maps cimport PreshMap
from preshed.maps cimport MapStruct
from preshed.maps cimport Cell

from .cache cimport ScoresCache

from .typedefs cimport *
from .features cimport Feature

from .sparse cimport SparseArrayC


cdef class LinearModel:
    cdef time_t time
    cdef readonly class_t nr_class
    cdef readonly int nr_templates
    cdef size_t n_corr
    cdef size_t total
    cdef Pool mem
    cdef PreshMap weights
    cdef PreshMap train_weights
    cdef ScoresCache cache
    cdef weight_t* scores

    cpdef int update(self, dict counts) except -1
    cdef const weight_t* get_scores(self, const Feature* feats, int n_feats) nogil
    cdef int set_scores(self, weight_t* scores, const Feature* feats, int n_feats) nogil


cdef class _Writer:
    cdef FILE* _fp
    cdef class_t _nr_class
    cdef count_t _freq_thresh

    cdef int write(self, feat_t feat_id, SparseArrayC* feat) except -1


cdef class _Reader:
    cdef FILE* _fp
    cdef class_t _nr_class
    cdef count_t _freq_thresh

    cdef int read(self, Pool mem, feat_t* out_id, SparseArrayC** out_feat) except -1
