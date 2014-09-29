from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue

from thinc.ml.learner cimport W as weight_t


ctypedef pair[size_t, size_t] Candidate
ctypedef pair[double, Candidate] Entry
ctypedef priority_queue[Entry] Queue


cdef class Move:
    cdef double score
    cdef size_t clas
    cdef size_t cost
    cdef Move prev


cdef class Beam:
    cdef size_t moves
    cdef size_t nr_class
    cdef size_t width
    cdef Queue q
    cdef list history
    cdef list extensions
    cdef list bests

    cdef int fill(self, double** scores)

    cpdef pair[size_t, size_t] pop(self) except *


cdef class MaxViolation:
    cdef weight_t delta
    cdef list pred
    cdef list gold

    cpdef weight_t check(self, weight_t p_score, weight_t g_score, list p_hist,
                          list g_hist) except -1
