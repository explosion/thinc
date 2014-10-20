cdef class Instance:
    def __init__(self, int n_context, int n_feats, int n_class,
                 clas=0, context=None, feats=None, values=None, scores=None):
        self.mem = Pool()
        self.n_context = n_context
        self.n_class = n_class
        self.n_feats = n_feats
        self.context = <size_t*>self.mem.alloc(n_context, sizeof(size_t))
        self.feats = <feat_t*>self.mem.alloc(n_feats, sizeof(feat_t))
        self.values = <weight_t*>self.mem.alloc(n_feats, sizeof(weight_t))
        self.scores = <weight_t*>self.mem.alloc(n_class, sizeof(weight_t))
        self.clas = clas
        cdef int i
        cdef size_t c
        if context is not None:
            for i, c in enumerate(context):
                self.context[i] = c
        cdef feat_t f
        if feats is not None:
            for i, f in enumerate(feats):
                self.feats[i] = f
        cdef weight_t v
        if values is not None:
            for i, v in enumerate(values):
                self.values[i] = v
        cdef weight_t s
        if scores is not None:
            for i, s in enumerate(scores):
                self.scores[i] = s

    cpdef class_t classify(self, LinearModel model, size_t[:] context=None,
                           feat_t[:] feats=None, Extractor extractor=None):
        cdef int i
        cdef size_t c
        if context is not None:
            for i, c in enumerate(context):
                self.context[i] = c
        if extractor is not None:
            extractor.extract(self.feats, self.context)
        cdef feat_t f
        if feats is not None:
            for i, f in enumerate(feats):
                self.feats[i] = f
        self.clas = model.score(self.scores, self.feats, self.values, self.n_feats)
        return self.clas

    property feats:
        def __get__(self):
            return [self.feats[i] for i in range(self.n_feats)]

        def __set__(self, list feats):
            assert len(feats) == self.n_feats
            for i, f in enumerate(feats):
                self.feats[i] = f

    property values:
        def __get__(self):
            return [self.values[i] for i in range(self.n_feats)]

        def __set__(self, list values):
            assert len(values) == self.n_feats
            for i, f in enumerate(values):
                self.values[i] = f

    property scores:
        def __get__(self):
            return [self.scores[i] for i in range(self.n_class)]

        def __set__(self, list scores):
            assert len(scores) == self.n_class
            for i, s in enumerate(scores):
                self.scores[i] = s
