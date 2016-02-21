cdef extern from "prng/normal.h":
    void normal_setup() nogil
    double normal() nogil
