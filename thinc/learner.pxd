from libc.stdio cimport FILE
from libc.string cimport memset

from cymem.cymem cimport Pool

from preshed.maps cimport PreshMap
from preshed.maps cimport MapStruct
from preshed.maps cimport Cell
from preshed.maps cimport map_get



from .cache cimport ScoresCache

from .typedefs cimport *
from .features cimport Extractor, Feature

from .sparse cimport SparseArrayC


cdef int arg_max(const weight_t* scores, const int n_classes) nogil

cdef int arg_max_if_true(const weight_t* scores, const int* is_valid,
                         const int n_classes) nogil

cdef int arg_max_if_zero(const weight_t* scores, const int* costs,
                         const int n_classes) nogil


cdef class LinearModel:
    cdef Extractor extractor
    cdef time_t time
    cdef readonly class_t n_classes
    cdef size_t n_corr
    cdef size_t total
    cdef Pool mem
    cdef PreshMap weights
    cdef PreshMap train_weights
    cdef ScoresCache cache
    cdef weight_t* scores
    cdef readonly int is_updating

    cdef void set_scores(self, weight_t* scores, const Feature* feats, int nr_feat) nogil

    cdef int update(self, const Feature* feats, int nr_feat, int best, int guess,
                    weight_t weight) except -1

    cdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1
 

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
