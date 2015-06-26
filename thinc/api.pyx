from libc.string cimport memset


cdef class Example:
    def __init__(self, int n_classes, int n_atoms):
        self.mem = Pool()

        self.n_classes = n_classes
        self.n_atoms = n_atoms
        
        self.atoms    = <atom_t*>  self.mem.alloc(n_atoms,   sizeof(atom_t))
        self.is_valid = <bint*>    self.mem.alloc(n_classes, sizeof(bint))
        self.costs    = <int*>     self.mem.alloc(n_classes, sizeof(int))
        self.scores   = <weight_t*>self.mem.alloc(n_classes, sizeof(weight_t))

        self.guess = 0
        self.best = 0
        self.cost = 0
        self.loss = 0

    def wipe(self):
        memset(self.atoms,    0, self.n_atoms   * sizeof(self.atoms[0]))
        memset(self.is_valid, 0, self.n_classes * sizeof(self.is_valid[0]))
        memset(self.costs,    0, self.n_classes * sizeof(self.costs[0]))
        memset(self.scores,   0, self.n_classes * sizeof(self.scores[0]))
