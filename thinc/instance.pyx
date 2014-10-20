cdef class Instance:
    def __init__(self, int n_context, int n_class, int n_feats):
        self.mem = Pool()
        self.context = <size_t*>self.mem.alloc(n_context, sizeof(size_t))
        self.feats = <feat_t*>self.mem.alloc(n_feats, sizeof(feat_t))
        self.values = <weight_t*>self.mem.alloc(n_feats, sizeof(weight_t))
        self.scores = <weight_t*>self.mem.alloc(n_class, sizeof(weight_t))
        self.clas = 0

    cpdef class_t classify(self, size_t[:] context, Extractor extractor, LinearModel model):
        cdef int i
        cdef size_t c
        for i, c in enumerate(context):
            self.context[i] = c
        extractor.extract(self.feats, self.context)
        self.clas = model.score(self.scores, self.feats, self.values, self.n_feats)
        return self.clas
