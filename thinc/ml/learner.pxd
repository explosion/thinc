from libc.stdint cimport uint64_t
from libc.stdint cimport uint16_t

from cymem.cymem cimport Pool

from preshed.maps cimport PreshMap
from preshed.maps cimport PreshMapArray
from preshed.maps cimport MapStruct
from preshed.maps cimport Cell


# Typedef numeric types, to make them easier to change and ensure consistency
ctypedef uint64_t F # Feature ID
ctypedef size_t C # Class
ctypedef double W # Weight
ctypedef size_t I # Index


# Number of weights in a line. Should be aligned to cache lines.
DEF LINE_SIZE = 7


# A set of weights, to be read in. Start indicates the class that w[0] refers
# to. Subsequent weights go from there.
cdef struct WeightLine:
    C start
    W[LINE_SIZE] line


cdef struct MetaData:
    W total
    I count
    I time
    

cdef struct TrainFeat:
    W** weights
    MetaData** meta


cdef class ScoresCache:
    cdef uint64_t i
    cdef uint64_t max_size
    cdef size_t scores_size
    cdef Pool _pool
    cdef W** _arrays
    cdef W* _scores_if_full
    cdef PreshMap _cache
    cdef size_t n_hit
    cdef size_t n_total

    cdef W* lookup(self, size_t size, void* kernel, bint* success)


cdef class LinearModel:
    cdef I time
    cdef C nr_class
    cdef I nr_templates
    cdef I n_corr
    cdef I total
    cdef Pool mem
    cdef PreshMapArray weights
    cdef PreshMapArray train_weights
    cdef ScoresCache cache
    cdef W* scores
    cdef WeightLine* _weight_lines

    cdef TrainFeat* new_feat(self, I template_id, F feat_id) except NULL
    cdef I gather_weights(self, WeightLine* w_lines, F* feat_ids, I nr_active) except *
    cdef int score(self, W* inplace, F* features, I nr_active) except -1
    cpdef int update(self, dict counts) except -1
