from libc.stdint cimport uint64_t, int64_t
from cymem.cymem cimport Pool


ctypedef size_t feat_t


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


cdef struct Feature:
    size_t* vals
    size_t n
    bint is_active


cdef class Extractor:
    cdef Pool mem
    cdef size_t nr_template
    cdef feat_t* features
    cdef Template* templates
    cdef readonly size_t nr_match
    cdef readonly size_t nr_feat
    cdef MatchPred* match_preds
    cdef feat_t* extract(self, size_t* context) except NULL
    cdef int count(self, dict counts, feat_t* features, double inc) except -1
