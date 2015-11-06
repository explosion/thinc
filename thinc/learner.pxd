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
