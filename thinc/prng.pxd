cdef extern from "prng/normal.h":
    void normal_setup() nogil
    double normal() nogil


cdef extern from "prng/MT19937.h":
    double uniform_double_PRN() nogil
