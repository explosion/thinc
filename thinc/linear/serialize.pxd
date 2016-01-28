from libc.stdio cimport FILE, fopen, fclose, fread, fwrite, feof, fseek
from libc.errno cimport errno
from ..typedefs cimport class_t, count_t, feat_t
from ..structs cimport SparseArrayC

from cymem.cymem cimport Pool


cdef class Writer:
    cdef FILE* _fp
    cdef class_t _nr_class
    cdef count_t _freq_thresh


    cdef int write(self, feat_t feat_id, SparseArrayC* feat) except -1


cdef class Reader:
    cdef FILE* _fp
    cdef class_t _nr_class
    cdef count_t _freq_thresh

    cdef int read(self, Pool mem, feat_t* out_id, SparseArrayC** out_feat) except -1
