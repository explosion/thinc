from libc.stdio cimport FILE, fopen, fclose, fread, fwrite, feof, fseek
from libc.errno cimport errno
from libc.stdint cimport int32_t
from ..typedefs cimport class_t, count_t, feat_t
from ..structs cimport SparseArrayC

from cymem.cymem cimport Pool


cdef class Writer:
    cdef FILE* _fp

    cdef int write(self, feat_t feat_id, SparseArrayC* feat) except -1


cdef class Reader:
    cdef FILE* _fp
    cdef public int32_t nr_feat

    cdef int read(self, Pool mem, feat_t* out_id, SparseArrayC** out_feat) except -1
