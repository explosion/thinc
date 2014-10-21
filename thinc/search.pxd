from cymem.cymem cimport Pool

from libc.stdint cimport uint32_t
from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector

from thinc.learner cimport weight_t
from thinc.learner cimport class_t


ctypedef pair[size_t, size_t] Candidate
ctypedef pair[weight_t, Candidate] Entry
ctypedef priority_queue[Entry] Queue


cdef class Beam:
    cdef Pool mem
    cdef class_t nr_class
    cdef class_t width
    cdef class_t size
    cdef Queue q
    cdef void** parents
    cdef void** states

    cdef int fill(self, weight_t** scores) except -1
    cpdef pair[size_t, size_t] pop(self) except *


cdef class MaxViolation:
    cdef int cost
    cdef weight_t delta
    cdef class_t n
    cdef void* pred
    cdef void* gold

    cdef weight_t check(self, int cost, weight_t p_score, weight_t g_score,
                         void* p, void* g, class_t n) except -1
