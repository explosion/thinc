from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue


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

    cdef int fill(self, double** scores, size_t** costs)

    cpdef pair[size_t, size_t] pop(self) except *
