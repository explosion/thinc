from libc.stdio cimport FILE
from libc.string cimport memset

from cymem.cymem cimport Pool

from preshed.maps cimport PreshMap
from preshed.maps cimport MapStruct
from preshed.maps cimport Cell

from .cache cimport ScoresCache

from .weights cimport WeightLine
from .weights cimport TrainFeat
from .typedefs cimport *
from .features cimport Feature
from .weights cimport gather_weights, set_scores


cdef class LinearModel:
    cdef time_t time
    cdef readonly class_t nr_class
    cdef readonly int nr_templates
    cdef size_t n_corr
    cdef size_t total
    cdef Pool mem
    cdef PreshMap weights
    cdef ScoresCache cache
    cdef weight_t* scores
    cdef WeightLine* _weight_lines
    cdef size_t _max_wl

    cpdef int update(self, dict counts) except -1

    cdef inline const weight_t* get_scores(self, const Feature* feats, const int n_feats) nogil:
        memset(self.scores, 0, self.nr_class * sizeof(weight_t))
        self.set_scores(self.scores, feats, n_feats)
        return self.scores

    cdef inline int set_scores(self, weight_t* scores, const Feature* feats, const int n_feats) nogil:
        cdef int f_i = gather_weights(self.weights.c_map, self.nr_class, self._weight_lines,
                             feats, n_feats)
        set_scores(scores, self._weight_lines, f_i, self.nr_class)
        return 0



cdef class _Writer:
    cdef FILE* _fp
    cdef class_t _nr_class
    cdef count_t _freq_thresh

    cdef int write(self, feat_t feat_id, TrainFeat* feat) except -1


cdef class _Reader:
    cdef FILE* _fp
    cdef class_t _nr_class
    cdef count_t _freq_thresh

    cdef int read(self, Pool mem, feat_t* out_id, TrainFeat** out_feat) except -1
