cdef struct Token:
    size_t lex
    size_t backoff
    size_t tag


cdef struct Dep:
    size_t label
    Token child


cdef struct Subtree:
    Dep a
    Dep b


cdef struct ParseContext:
    Token s0
    Token s1
    Token s2
    Token n0
    Token n1
    Token n2

    Subtree n0L

    Subtree s0L
    Subtree s0R
    Subtree s1L
    Subtree s1R
    Subtree s2L
    Subtree s2R
