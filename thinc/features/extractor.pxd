from libc.stdint cimport uint32_t

from cymem.cymem cimport Pool
from preshed.tries cimport SequenceIndex

ctypedef uint32_t feat_t
ctypedef uint32_t idx_t
ctypedef uint32_t context_t


DEF MAX_FEAT_LEN = 10


cdef struct Template:
    size_t id
    size_t n
    context_t[MAX_FEAT_LEN] raws
    idx_t[MAX_FEAT_LEN] args


cdef struct MatchPred:
    size_t id
    idx_t idx1
    idx_t idx2
    context_t[2] raws


cdef class Extractor:
    cdef Pool mem
    cdef SequenceIndex trie
    cdef readonly size_t nr_template
    cdef Template* templates
    cdef feat_t* _features
    cdef readonly size_t nr_match
    cdef readonly size_t nr_feat
    cdef MatchPred* match_preds
    cdef feat_t* extract(self, size_t* context) except NULL
    cdef int count(self, dict counts, feat_t* features, double inc) except -1
