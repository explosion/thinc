from ..structs cimport MinibatchC


cdef class Minibatch:
    cdef MinibatchC* c

    @staticmethod
    cdef inline take_ownership(MinibatchC* mb):
        cdef Minibatch self = Minibatch.__new__(Minibatch)
        self.c = mb
        return self
