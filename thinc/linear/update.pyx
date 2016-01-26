cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc

from .api cimport Example
from .typedefs cimport time_t, feat_t, weight_t, class_t
from .structs cimport SparseAverageC
from .sparse cimport SparseArray


cdef class Updater:
    def __call__(self, Example eg):
        raise NotImplementedError

    cdef void update(self, ExampleC* eg) except *:
        raise NotImplementedError


cdef class AveragedPerceptronUpdater:
    def __init__(self, PreshMap weights):
        self.time = 0
        self.train_weights = PreshMap()
        self.weights = weights
        self.mem = Pool()


