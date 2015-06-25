from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc


cdef class SparseArray:
    def __init__(self, int clas, weight_t value):
        self.c = init(clas, value)

    def __dealloc__(self):
        PyMem_Free(self.c)

    def __getitem__(self, int key):
        cdef int i = find_key(self.c, key)
        if i >= 0:
            return self.c[i].val
        else:
            return 0

    def __setitem__(self, int key, weight_t value):
        cdef int i = find_key(self.c, key)
        if i < 0:
            self.c = resize(self.c)
            i = find_key(self.c, key)
        self.c[i] = SparseArrayC(key=key, val=value)

    def __iter__(self):
        cdef int i = 0
        while self.c[i].key >= 0:
            yield (self.c[i].key, self.c[i].val)
            i += 1


cdef SparseArrayC* init(int key, weight_t value) except NULL:
    array = <SparseArrayC*>PyMem_Malloc(3 * sizeof(SparseArrayC))
    array[0] = SparseArrayC(key=key, val=value)
    array[1] = SparseArrayC(key=-1, val=0)
    array[2] = SparseArrayC(key=-2, val=0)
    return array


cdef int find_key(const SparseArrayC* array, int key) except -2:
    cdef int i = 0
    while array[i].key != -2:
        if array[i].key == key:
            return i
        elif array[i].key == -1:
            return i
        else:
            i += 1
    else:
        return -1


cdef SparseArrayC* resize(SparseArrayC* array) except NULL:
    cdef int length = 0
    while array[length].key != -2:
        length += 1
    new_length = length * 2
    array = <SparseArrayC*>PyMem_Realloc(array, new_length * sizeof(SparseArrayC))
    cdef int i
    for i in range(length, new_length-1):
        array[i] = SparseArrayC(key=-1, val=0)
    array[new_length-1] = SparseArrayC(key=-2, val=0)
    return array
