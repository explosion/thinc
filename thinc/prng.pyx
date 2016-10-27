cdef extern from "prng/normal.h":
    void normal_setup() nogil
    double normal() nogil


cdef extern from "prng/MT19937.h":
    double uniform_double_PRN() nogil
    void mt_init() nogil

normal_setup()
mt_init()

def get(double loc=0, double scale=1):
    return normal() * scale + loc


cdef double get_uniform() nogil:
    return uniform_double_PRN()

cdef double get_normal() nogil:
    return uniform_double_PRN()

