from cymem.cymem cimport Pool
from ..structs cimport ExampleC, FeatureC, MapC
from ..typedefs cimport feat_t, weight_t, atom_t


cdef class Example:
    cdef Pool mem
    cdef ExampleC* c

    @staticmethod
    cdef inline Example from_ptr(ExampleC* ptr):
        cdef Example eg = Example.__new__(Example)
        eg.c = ptr
        return eg
