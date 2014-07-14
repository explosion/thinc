ctypedef size_t I


cdef struct Node:
    size_t value
    size_t first
    size_t last
    Node* nodes


cdef class AddressTree:
    cdef Node* tree
    cdef int lookup(self, size_t* feature, size_t n) except -1
    cdef int insert(self, size_t value, size_t* feature, size_t n) except -1
