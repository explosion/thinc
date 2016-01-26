cdef class AveragedPerceptron:
    def __init__(self):
        self.weights = PreshMap()
        self.time = 0
        self.averages = PreshMap()
        self.mem = Pool()

    def __dealloc__(self):
        cdef size_t feat_addr
        # Use 'raw' memory management, instead of cymem.Pool, for weights.
        # The memory overhead of cymem becomes significant here.
        if self.weights is not None:
            for feat_addr in self.weights.values():
                if feat_addr != 0:
                    PyMem_Free(<SparseArrayC*>feat_addr)
        if self.averages is not None:
            for feat_addr in self.train_weights.values():
                if feat_addr != 0:
                    feat = <SparseAverageC*>feat_addr
                    PyMem_Free(feat.avgs)
                    PyMem_Free(feat.times)

    def __call__(self, Example eg):
        self.extractor.set_features(eg.c.features, eg.c.atoms)
        self.set_scores(eg.c.scores, eg.c.features, eg.c.nr_feat)
        #eg.c.guess = arg_max_if_true(eg.c.scores, eg.c.is_valid, eg.c.nr_class)
        PyErr_CheckSignals()

    def dump(self, loc):
        cdef Writer writer = Writer(loc)
        for i, (key, feat_addr) in enumerate(self.weights.items()):
            if feat_addr != 0:
                writer.write(key, <SparseArrayC*>feat_addr)
            if i % 1000 == 0:
                PyErr_CheckSignals()
        writer.close()

    def load(self, loc):
        cdef feat_t feat_id
        cdef SparseArrayC* feature
        cdef Reader reader = Reader(loc)
        cdef int i = 0
        while reader.read(self.mem, &feat_id, &feature):
            self.weights.set(feat_id, feature)
            if i % 1000 == 0:
                PyErr_CheckSignals()
            i += 1

    def end_training(self):
        cdef feat_id
        cdef size_t feat_addr
        cdef int i = 0
        for feat_id, feat_addr in self.train_weights.items():
            if feat_addr != 0:
                feat = <SparseArrayC*>feat_addr
                while feat.curr[i].key >= 0:
                    unchanged = (time + 1) - <time_t>feat.times[i].val
                    feat.avgs[i].val += unchanged * feat.curr[i].val
                    feat.curr[i].val = feat.avgs[i].val / time
                    i += 1

    cdef void set_scores(self, weight_t* scores, const FeatureC* feats, int nr_feat) nogil:
        # This is the main bottle-neck of spaCy --- where we spend all our time.
        # Typical sizes for the dependency parser model:
        # * weights_table: ~9 million entries
        # * n_feats: ~200
        # * scores: ~80 classes
        # 
        # I think the bottle-neck is actually reading the weights from main memory.
        cdef const MapStruct* weights_table = self.weights.c_map
        cdef int i, j
        cdef FeatureC feat
        for i in range(nr_feat):
            feat = feats[i]
            class_weights = <const SparseArrayC*>map_get(weights_table, feat.key)
            if class_weights != NULL:
                j = 0
                while class_weights[j].key >= 0:
                    scores[class_weights[j].key] += class_weights[j].val * feat.value
                    j += 1

    @cython.cdivision(True)
    cdef void update(self, ExampleC* eg) except *:
        self.time += 1
        if eg.costs[eg.guess] != 0:
            for feat in eg.features[:eg.nr_feat]:
                self.update_weight(feat.key, eg.best, feat.value * eg.costs[eg.guess])
                self.update_weight(feat.key, eg.guess, -feat.value * eg.costs[eg.guess])

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        if upd == 0:
            return
        feat = <SparseAverageC*>self.train_weights.get(feat_id)
        if feat == NULL:
            feat = <SparseAverageC*>PyMem_Malloc(sizeof(SparseAverageC))
            feat.curr  = SparseArray.init(clas, weight)
            feat.avgs  = SparseArray.init(clas, 0)
            feat.times = SparseArray.init(clas, <weight_t>time)
            self.train_weights.set(feat_id, feat)
            self.weights.set(feat_id, feat.curr)
        else:  
            i = SparseArray.find_key(feat.curr, key)
            if i <= 0:
                feat.curr = SparseArray.resize(feat.curr)
                feat.avgs = SparseArray.resize(feat.avgs)
                feat.times = SparseArray.resize(feat.times)
                self.weights.set(feat_id, feat.curr)
                i = SparseArray.find_key(feat.curr, key)
            feat.curr[i].key = key
            feat.avgs[i].key = key
            # Apply the last round of updates, multiplied by the time unchanged
            feat.avgs[i].val += (time - feat.times[i].val) * feat.curr[i].val
            feat.curr[i].val += upd
            feat.times[i].val = time
