from libc.string cimport memset


cdef class Example:
    def __init__(self, int n_classes, int n_atoms, int n_embeddings):
        self.mem = Pool()

        self.n_classes = n_classes
        self.n_atoms = n_atoms
        self.n_features = n_embeddings
        
        self.atoms      = <atom_t[:n_atoms]>self.mem.alloc(n_atoms, sizeof(atom_t))
        self.is_valid   = <int[:n_classes]>self.mem.alloc(n_classes, sizeof(int))
        self.costs      = <int[:n_classes]>self.mem.alloc(n_classes, sizeof(int))
        self.scores     = <weight_t[:n_classes]>self.mem.alloc(n_classes, sizeof(weight_t))
        self.embeddings = <weight_t[:n_embeddings]>self.mem.alloc(n_embeddings, sizeof(float))

        self.guess = 0
        self.best = 0
        self.cost = 0
        self.loss = 0

    def wipe(self):
        cdef int i
        for i in range(self.n_classes):
            self.is_valid[i] = 0
            self.costs[i] = 0
            self.scores[i] = 0
        for i in range(self.n_atoms):
            self.atoms[i] = 0
        for i in range(self.n_features):
            self.embeddings[i] = 0
