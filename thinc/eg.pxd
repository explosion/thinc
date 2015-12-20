from cymem.cymem cimport Pool
from libc.string cimport memset

from .structs cimport ExampleC, BatchC, FeatureC
from .typedefs cimport weight_t, atom_t


cdef class Example:
    cdef Pool mem
    cdef ExampleC c


cdef class Batch:
    cdef Pool mem
    cdef BatchC c
