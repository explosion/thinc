cdef class Instance:
    def __init__(self, int n_atoms, int n_feats, int n_classes):
        self.mem = Pool()
        self.n_atoms = n_atoms
        self.n_classes = n_classes
        self.n_feats = n_feats
        self.atoms = <size_t*>self.mem.alloc(n_atoms, sizeof(size_t))
        self.feats = <feat_t*>self.mem.alloc(n_feats, sizeof(feat_t))
        self.values = <weight_t*>self.mem.alloc(n_feats, sizeof(weight_t))
        self.scores = <weight_t*>self.mem.alloc(n_classes, sizeof(weight_t))
        self.clas = 0

    def extract(self, size_t[:] atoms, Extractor extractor):
        cdef int i
        cdef size_t a
        for i, atom in enumerate(atoms):
            self.atoms[i] = atom
        extractor.extract(self.feats, self.atoms)

    def predict(self, LinearModel model):
        self.clas = model.score(self.scores, self.feats, self.values)
        return self.clas

    property feats:
        def __get__(self):
            return [self.feats[i] for i in range(self.n_feats)]

        def __set__(self, list feats):
            for i, f in enumerate(feats):
                self.feats[i] = f

    property values:
        def __get__(self):
            return [self.values[i] for i in range(self.n_feats)]

        def __set__(self, list values):
            for i, f in enumerate(values):
                self.values[i] = f

    property scores:
        def __get__(self):
            return [self.scores[i] for i in range(self.n_class)]

        def __set__(self, list scores):
            assert len(scores) == self.n_classes
            for i, s in enumerate(scores):
                self.scores[i] = s
