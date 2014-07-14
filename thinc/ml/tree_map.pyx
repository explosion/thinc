from libc.stdlib cimport calloc, malloc, free


cdef size_t* array_from_tuple(t):
    cdef size_t length = len(t)
    cdef size_t* a = <size_t*>calloc(length, sizeof(size_t))
    cdef size_t i, v
    for i in range(length):
        v = t[i]
        a[i] = v
    return a


cdef class AddressTree:
    def __cinit__(self):
        self.tree = <Node*>calloc(1, sizeof(Node))
        assert self.tree.nodes is NULL

    def __dealloc__(self):
        free_node(self.tree)

    property first:
        def __get__(self):
            return self.tree.first

    property last:
        def __get__(self):
            return self.tree.last

    property value:
        def __get__(self):
            return self.tree.nodes[0].value

    def lookup_tuple(self, feature):
        cdef size_t* c_feat = array_from_tuple(feature)
        cdef size_t value = self.lookup(c_feat, len(feature))
        free(c_feat)
        return value

    def insert_tuple(self, feature, value):
        cdef size_t* c_feat = array_from_tuple(feature)
        self.insert(value, c_feat, len(feature))
        free(c_feat)

    cdef int insert(self, size_t value, size_t* feature, size_t n) except -1:
        cdef Node* node = self.tree
        i = 0
        for i in range(n):
            # Deepen tree
            if node.nodes is NULL:
                node.nodes = <Node*>calloc(1, sizeof(Node))
                node.first = feature[i]
                node.last = feature[i] + 1
                node = &node.nodes[0]
            # Broaden branch
            elif feature[i] < node.first or feature[i] >= node.last:
                first = min(feature[i], node.first)
                last = max(feature[i] + 1, node.last)
                new = <Node*>calloc(last - first, sizeof(Node))
                for j in range(node.last - node.first):
                    new[j] = node.nodes[j]
                free(node.nodes)
                node.nodes = new
                node.first = first
                node.last = last
                node = &node.nodes[feature[i] - node.first]
            else:
                node = &node.nodes[feature[i] - node.first]
        node.value = value

    cdef int lookup(self, size_t* feature, I n) except -1:
        cdef Node* node = self.tree
        cdef size_t i
        for i in range(n):
            if node.nodes is NULL:
                return 0
            elif feature[i] < node.first or feature[i] >= node.last:
                return 0
            node = &node.nodes[feature[i] - node.first]
        return node.value





cdef int free_node(Node* node):
    if node == NULL:
        return 0

    for i in range(node.first, node.last):
        free_node(node)
