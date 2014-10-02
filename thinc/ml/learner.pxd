from libc.stdint cimport uint64_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uint16_t

from cymem.cymem cimport Pool

from preshed.maps cimport PreshMap
from preshed.maps cimport PreshMapArray
from preshed.maps cimport MapStruct
from preshed.maps cimport Cell


#DEF WEIGHT_TYPE = 'float'
#
#IF WEIGHT_TYPE == 'float':
#ELIF WEIGHT_TYPE == 'double':
#    ctypedef double weight_t
#ELIF WEIGHT_TYPE == 'int':
#    ctypedef int weight_t
#ELSE:
#    ctypedef double weight_t
ctypedef int weight_t

# Typedef numeric types, to make them easier to change and ensure consistency
ctypedef uint64_t feat_t
ctypedef uint32_t class_t
ctypedef uint32_t count_t
ctypedef uint32_t time_t


# Number of weights in a line. Should be aligned to cache lines.
DEF LINE_SIZE = 8

ctypedef weight_t[LINE_SIZE] weight_line_t


# A set of weights, to be read in. Start indicates the class that w[0] refers
# to. Subsequent weights go from there.
cdef struct WeightLine:
    int start
    weight_line_t line


cdef struct MetaData:
    weight_t total
    count_t count
    time_t time
    

cdef struct TrainFeat:
    size_t length
    WeightLine** weights
    MetaData** meta


cdef class ScoresCache:
    cdef size_t i
    cdef size_t max_size
    cdef class_t scores_size
    cdef Pool _pool
    cdef weight_t** _arrays
    cdef weight_t* _scores_if_full
    cdef PreshMap _cache
    cdef size_t n_hit
    cdef size_t n_total

    cdef weight_t* lookup(self, class_t size, void* kernel, bint* success)


cdef class LinearModel:
    cdef time_t time
    cdef readonly class_t nr_class
    cdef size_t nr_templates
    cdef size_t n_corr
    cdef size_t total
    cdef Pool mem
    cdef PreshMapArray weights
    cdef ScoresCache cache
    cdef weight_t* scores
    cdef WeightLine** _weight_lines

    cdef TrainFeat* new_feat(self, size_t template_id, feat_t feat_id) except NULL
    cdef int score(self, weight_t* inplace, feat_t* features, size_t nr_active) except -1
    cpdef int update(self, dict counts) except -1
