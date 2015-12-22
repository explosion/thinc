from cymem.cymem cimport Pool

from .typedefs cimport weight_t, atom_t

from preshed.maps cimport PreshMap


cdef struct Param:
    void update(Param* self, float* gradient, int t, float eta, float mu) except *
    float* curr
    float* avg
    float* step
    int length


cdef class EmbeddingTable:
    cdef Pool mem
    cdef public object initializer
    cdef readonly PreshMap table
    cdef readonly int n_cols
 
    cdef Param* get(self, atom_t key) except NULL


cdef class InputLayer:
    cdef Pool mem

    cdef int length
    cdef readonly list indices
    cdef readonly list tables
