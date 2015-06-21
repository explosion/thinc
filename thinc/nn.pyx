cimport numpy as np
cimport cython

from cymem.cymem cimport Pool

from .typedefs cimport weight_t

from libc.string cimport memcpy
 

cdef class EmbeddingTable:
    cdef Pool mem
    cdef weight_t** rows
    cdef weight_t* _buffer
    cdef readonly int n_rows
    cdef readonly int n_cols
    cdef int _slice_size
    def __init__(self, n_rows, n_cols, slice_size=4, get_value=None):
        n_rows += 1
        n_cols += 1
        mem = Pool()
        rows = <weight_t**>mem.alloc(n_rows, sizeof(weight_t*))

        cdef int i, j
        for i in range(n_rows):
            rows[i] = <weight_t*>mem.alloc(n_cols, sizeof(weight_t))

        if get_value is not None:
            for i in range(n_rows):
                for j in range(n_cols):
                    rows[i][j] = get_value(i, j)

        slice_buffer = <weight_t*>mem.alloc(slice_size * n_cols, sizeof(weight_t))

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.mem = mem
        self.rows = rows
        self._buffer = slice_buffer
        self._slice_size = slice_size

    @cython.boundscheck(False)
    def inc_row(self, int idx, weight_t[:] updates):
        cdef int i
        cdef weight_t upd
        for i, upd in enumerate(updates):
            self.rows[idx][i] += upd


cdef class InputLayer:
    '''An input layer to an NN.'''
    cdef Pool mem

    cdef int length
    cdef readonly list lengths
    cdef readonly list tables
    cdef weight_t* _buffer

    def __init__(self, lengths, tables):
        self.length = sum(n * t.n_cols for (n, t) in zip(lengths, tables))
        self.lengths = list(lengths)
        self.tables = list(tables)

    def __len__(self):
        return self.length
    
    @cython.boundscheck(False)
    def fill(self, weight_t[:] output, slices):
        cdef int i, j, idx, c
        cdef EmbeddingTable table
        c = 0
        for i, (indices, table) in enumerate(zip(slices, self.tables)):
            for idx in indices:
                for j in range(table.n_cols):
                    output[c] = table.rows[idx][j]
                    c += 1

    @cython.boundscheck(False)
    def update(self, weight_t[:] update, slices):
        cdef int i, j, idx, c
        cdef EmbeddingTable table
        c = 0
        for i, (indices, table) in enumerate(zip(slices, self.tables)):
            for idx in indices:
                for j in range(table.n_cols):
                    table.rows[idx][j] += update[c]
                    c += 1
