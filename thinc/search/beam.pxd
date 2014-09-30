from cymem.cymem cimport Pool

from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector

from thinc.ml.learner cimport W as weight_t


ctypedef pair[size_t, size_t] Candidate
ctypedef pair[double, Candidate] Entry
ctypedef priority_queue[Entry] Queue


cdef class Beam:
    cdef Pool mem
    cdef size_t nr_class
    cdef size_t width
    cdef size_t size
    cdef Queue q
    cdef void** parents
    cdef void** states

    cdef int fill(self, double** scores) except -1
    cpdef pair[size_t, size_t] pop(self) except *


cdef class MaxViolation:
    cdef int cost
    cdef weight_t delta
    cdef size_t n
    cdef void* pred
    cdef void* gold

    cdef weight_t check(self, int cost, weight_t p_score, weight_t g_score,
                         void* p, void* g, size_t n) except -1
