from libc.stdlib cimport malloc, free


cpdef ParseContext get_field_map():
    cdef size_t* memory = alloc_sequential_size_t(sizeof(ParseContext))
    cdef ParseContext parse = (<ParseContext*>memory)[0]
    free(memory)
    return parse


# TODO: Is this a bad idea?
cdef size_t* alloc_sequential_size_t(size_t total_size):
    cdef size_t i = 0
    cdef size_t* memory = <size_t*>malloc(total_size)
    for i in range(sizeof(ParseContext) / sizeof(size_t)):
        memory[i] = i
    return memory


"""
cdef ParseContext seq_parse(size_t* array):
    cdef ParseContext p = ParseContext()
    p.s0 = seq_token(i)
    p.s1 = seq_token(i)
    p.s2 = seq_token(i)
    p.n0 = seq_token(i)
    p.n1 = seq_token(i)
    p.n2 = seq_token(i)

    p.n0L = seq_subtree(i)

    p.s0L = seq_subtree(i)
    p.s0R = seq_subtree(i)
    p.s1L = seq_subtree(i)
    p.s1R = seq_subtree(i)
    p.s2L = seq_subtree(i)
    p.s2R = seq_subtree(i)
    return p


cdef Token seq_token(size_t* i):
    cdef Token t = Token()
    t.lex = i[0]
    i[0] += 1
    t.backoff = i[0]
    i[0] += 1
    t.tag = i[0]
    i[0] += 1
    return t

cdef Subtree seq_subtree(size_t* i):
    cdef Subtree st = Subtree()
    st.a = seq_dep(i)
    st.b = seq_dep(i)
    return st

cdef Dep seq_dep(size_t* i):
    cdef Dep dep = Dep()
    dep.label = i[0]
    i[0] += 1
    dep.child = seq_token(i)
    return dep
"""
