from cpython.mem cimport PyMem_Malloc
from cpython.exc cimport PyErr_SetFromErrno

from libc.stdio cimport FILE, fopen, fclose, fread, fwrite, feof, fseek

from libc.stdlib cimport qsort
from libc.stdint cimport int32_t

from .sparse cimport SparseArray

from ..structs cimport SparseArrayC
from ..typedefs cimport feat_t

from os import path


cdef class Writer:
    def __init__(self, object loc, int32_t nr_feat):
        if path.exists(loc):
            assert not path.isdir(loc)
        cdef bytes bytes_loc = loc.encode('utf8') if type(loc) == unicode else loc
        self._fp = fopen(<char*>bytes_loc, 'wb')
        assert self._fp != NULL
        fseek(self._fp, 0, 0)
        # Write a 32 bit int to the file representing the number of features.
        # Before v0.100, this was the number of classes. Now it's the *total*
        # length of the hash table, including empty slots. This ensures the hash
        # table never has to be resized, speeding loading up by 10x
        _write(&nr_feat, sizeof(nr_feat), 1, self._fp)

    def close(self):
        cdef size_t status = fclose(self._fp)
        assert status == 0

    cdef int write(self, feat_t feat_id, SparseArrayC* feat) except -1:
        if feat == NULL:
            return 0
        
        _write(&feat_id, sizeof(feat_id), 1, self._fp)
        
        cdef int i = 0
        while feat[i].key >= 0:
            i += 1
        cdef int32_t length = i
        
        _write(&length, sizeof(length), 1, self._fp)
        
        qsort(feat, length, sizeof(SparseArrayC), SparseArray.cmp)
        
        for i in range(length):
            _write(&feat[i].key, sizeof(feat[i].key), 1, self._fp)
            _write(&feat[i].val, sizeof(feat[i].val), 1, self._fp)


cdef int _write(void* value, size_t size, int n, FILE* fp) except -1:
    status = fwrite(value, size, 1, fp)
    assert status == 1, status


cdef packed struct _header_t:
    # row header as defined by model file format
    feat_t feat_id
    int32_t length


cdef class Reader:
    def __init__(self, loc):
        assert path.exists(loc)
        assert not path.isdir(loc)
        cdef bytes bytes_loc = loc.encode('utf8') if type(loc) == unicode else loc
        self._fp = fopen(<char*>bytes_loc, 'rb')
        if not self._fp:
            PyErr_SetFromErrno(IOError)
        status = fseek(self._fp, 0, 0)
        status = fread(&self.nr_feat, sizeof(self.nr_feat), 1, self._fp)
        if status < 1:
            raise IOError("empty input file" if feof(self._fp) else "error reading input file")
        # TODO: Remove this hack once users have migrated away from the v0.100.2
        # spaCy data. This hack allows previously distributed data to load quickly.
        # In previous versions, the initial 32 bit int at the start of the model
        # represented the number of classes. Now we use it to represent the
        # number of features, so that the hash table can be resized.
        # The hack here hard-codes the mapping from the number of classes in
        if self.nr_feat == 92:
            self.nr_feat = 16777216

    def __dealloc__(self):
        fclose(self._fp)

    cdef int read(self, Pool mem, feat_t* out_id, SparseArrayC** out_feat) except -1:
        cdef _header_t header
        status = fread(&header, sizeof(header), 1, self._fp)
        if status < 1:
            if feof(self._fp):
                return 0  # end of file
            raise IOError("error reading input file")
        feat = <SparseArrayC*>PyMem_Malloc((header.length + 1) * sizeof(SparseArrayC))
        if not feat:
            raise MemoryError()

        status = fread(feat, sizeof(SparseArrayC), header.length, self._fp)
        if status != <size_t> header.length:
            raise IOError("error reading input file")

        # Trust We allocated correctly above
        feat[header.length].key = -2 # Indicates end of memory region
        feat[header.length].val = 0

        # Copy into the output variables
        out_feat[0] = feat
        out_id[0] = header.feat_id
        # Signal whether to continue reading, to the outer loop
        if feof(self._fp):
            return 0
        else:
            return 1
