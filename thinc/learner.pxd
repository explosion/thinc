from libc.stdio cimport FILE
from libc.string cimport memset

from cymem.cymem cimport Pool

from preshed.maps cimport PreshMap
from preshed.maps cimport MapStruct
from preshed.maps cimport Cell
from preshed.maps cimport map_get



from .cache cimport ScoresCache

from .typedefs cimport *
from .features cimport Feature

from .sparse cimport SparseArrayC


cdef class LinearModel:
    cdef object extractor
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

    cdef int update(self, const Feature* feats, int nr_feat, int best, int guess,
                    weight_t weight) except -1

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1
 
    @staticmethod
    cdef inline void set_scores(weight_t* scores, const MapStruct* weights_table,
                         const Feature* feats, int nr_feat) nogil:
        # This is the main bottle-neck of spaCy --- where we spend all our time.
        # Typical sizes for the dependency parser model:
        # * weights_table: ~9 million entries
        # * n_feats: ~200
        # * scores: ~80 classes
        # 
        # I think the bottle-neck is actually reading the weights from main memory.
 
        cdef int i, j
        cdef Feature feat
        for i in range(nr_feat):
            feat = feats[i]
            class_weights = <const SparseArrayC*>map_get(weights_table, feat.key)
            if class_weights != NULL:
                j = 0
                while class_weights[j].key >= 0:
                    scores[class_weights[j].key] += class_weights[j].val * feat.value
                    j += 1


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
