from cymem.cymem cimport Pool
from libc.math cimport sqrt as c_sqrt
from libc.string cimport memset, memcpy, memmove

from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get

from ..structs cimport ExampleC, FeatureC, MapC
from ..typedefs cimport feat_t, weight_t, atom_t
from ..linalg cimport Vec, VecVec


cdef class Example:
    cdef Pool mem
    cdef ExampleC c

    @staticmethod
    cdef inline Example from_ptr(ExampleC* ptr):
        cdef Example eg = Example.__new__(Example)
        eg.c = ptr[0]
        return eg
