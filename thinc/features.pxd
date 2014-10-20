from libc.stdint cimport uint64_t, int64_t
from cymem.cymem cimport Pool


DEF MAX_FEAT_LEN = 10


cdef struct Template:
    size_t id
    size_t n
    uint64_t[MAX_FEAT_LEN] raws
    size_t[MAX_FEAT_LEN] args


cdef struct MatchPred:
    size_t id
    size_t idx1
    size_t idx2
    size_t[2] raws


cdef class Extractor:
    cdef Pool mem
    cdef size_t nr_template
    cdef Template* templates
    cdef readonly size_t nr_match
    cdef readonly size_t nr_feat
    cdef MatchPred* match_preds
    cdef int extract(self, uint64_t* features, size_t* context) except -1
    cdef int count(self, dict counts, uint64_t* features, double inc) except -1
