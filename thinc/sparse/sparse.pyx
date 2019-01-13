# cython: infer_types=True
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
from ..structs cimport SparseArrayC


cdef class SparseArray:
    def __init__(self, int clas, weight_t value):
        self.c = SparseArray.init(clas, value)

    def __dealloc__(self):
        PyMem_Free(self.c)

    def __getitem__(self, int key):
        cdef int i = SparseArray.find_key(self.c, key)
        if i >= 0:
            return self.c[i].val
        else:
            return 0

    def __setitem__(self, int key, weight_t value):
        cdef int i = SparseArray.find_key(self.c, key)
        if i < 0:
            self.c = SparseArray.resize(self.c)
            i = SparseArray.find_key(self.c, key)
        self.c[i] = SparseArrayC(key=key, val=value)

    def __iter__(self):
        cdef int i = 0
        while self.c[i].key >= 0:
            yield (self.c[i].key, self.c[i].val)
            i += 1

    def __lshift__(SparseArray self, SparseArray other):
        PyMem_Free(self.c)
        self.c = SparseArray.clone(other.c)
