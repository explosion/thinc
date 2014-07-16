class Token(ContextFields):
    pass

class Dep(ContextFields):
    pass

class Parse(ContextFields):
    s0 = Token
    s1 = Token
    s2 = Token
    n0 = Token
    n1 = Token
    n2 = Token

    s0L = Subtree
    s0L2 = Subtree
    s0R = Subtree
    s0R2 = Subtree
    s1L = Subtree
    s1L2 = Subtree
    s1
