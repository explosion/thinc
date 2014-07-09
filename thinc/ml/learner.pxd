from libc.stdint cimport uint64_t
from libc.stdint cimport uint16_t
from thinc.ext.sparsehash cimport dense_hash_map


# Typedef numeric types, to make them easier to change and ensure consistency
ctypedef uint64_t F # Feature ID
ctypedef uint16_t C # Class
ctypedef double W # Weight
ctypedef size_t I # Index


# Number of weights in a line. Should be aligned to cache lines.
DEF LINE_SIZE = 7


# A set of weights, to be read in. Start indicates the class that w[0] refers
# to. Subsequent weights go from there.
cdef struct WeightLine:
    C start
    W[LINE_SIZE] line


cdef struct CountLine:
    C start
    I[LINE_SIZE] line


cdef struct TrainFeat:
    WeightLine** weights
    WeightLine** totals
    CountLine** counts
    CountLine** times


cdef class LinearModel:
    cdef dense_hash_map[F, size_t] weights
    cdef dense_hash_map[F, size_t] metadata

    cdef I gather_weights(self, WeightLine* w_lines, F* feat_ids, I nr_active)
    cdef int score(self, W* inplace, F* features, I nr_active) except -1
    cdef int update(self, dict counts) except -1
