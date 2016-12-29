from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
from libc.stdint cimport int32_t
from ..typedefs cimport weight_t
from ..structs cimport SparseArrayC


cdef class SparseArray:
    cdef SparseArrayC* c

    @staticmethod
    cdef inline SparseArrayC* init(int key, weight_t value) except NULL:
        array = <SparseArrayC*>PyMem_Malloc(3 * sizeof(SparseArrayC))
        array[0] = SparseArrayC(key=key, val=value)
        array[1] = SparseArrayC(key=-1, val=0) # -1 marks end of values
        array[2] = SparseArrayC(key=-2, val=0) # -2 marks end of memory
        return array

    @staticmethod
    cdef inline int find_key(const SparseArrayC* array, int key) except -2:
        # Find either the key, or a place to insert it.
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
    
    @staticmethod
    cdef inline SparseArrayC* resize(SparseArrayC* array) except NULL:
        cdef int length = 0
        while array[length].key != -2:
            length += 1
        cdef int new_length = length * 2
        array = <SparseArrayC*>PyMem_Realloc(array, new_length * sizeof(SparseArrayC))
        cdef int i
        for i in range(length, new_length-1):
            array[i] = SparseArrayC(key=-1, val=0)
        array[new_length-1] = SparseArrayC(key=-2, val=0)
        return array

    @staticmethod
    cdef inline int cmp(const void* a, const void* b) nogil:
        # Three-way comparison (I always forget):
        # 0 if equal
        # -1 if a before b
        # 1 if b before a 
        cdef int key_a = (<SparseArrayC*>a).key
        cdef int key_b = (<SparseArrayC*>b).key
        if key_a == key_b:
            return 0
        elif key_a < 0 or key_b < 0:
            return -1 if a > b else 1 # Sort negatives to the end 
        else:
            return -1 if a < b else 1
