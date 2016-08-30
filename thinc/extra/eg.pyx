# cython: infer_types=True
cimport cython

cdef ExampleC* init_eg(Pool mem, int nr_class=0, int nr_atom=0, int nr_feat=0, widths=None):
    if widths is None:
        widths = [nr_class]
    if nr_class == 0:
        nr_class = widths[-1]

    eg = <ExampleC*>mem.alloc(1, sizeof(ExampleC))
    eg.nr_class = nr_class
    eg.nr_atom = nr_atom
    eg.nr_feat = nr_feat

    eg.scores = <weight_t*>mem.alloc(nr_class, sizeof(eg.scores[0]))
    eg.costs = <weight_t*>mem.alloc(nr_class, sizeof(eg.costs[0]))
    eg.atoms = <atom_t*>mem.alloc(nr_atom, sizeof(eg.atoms[0]))
    eg.features = <FeatureC*>mem.alloc(nr_feat, sizeof(eg.features[0]))
        
    eg.is_valid = <int*>mem.alloc(nr_class, sizeof(eg.is_valid[0]))
    for i in range(eg.nr_class):
        eg.is_valid[i] = 1
    return eg
    

cdef class Example:
    def __init__(self, int nr_class=0, int nr_atom=0, int nr_feat=0):
        self.mem = Pool()
        self.c = init_eg(self.mem, nr_class=nr_class, nr_atom=nr_atom, nr_feat=nr_feat)

    def fill_features(self, int value, int nr_feat):
        for i in range(nr_feat):
            self.c.features[i].i = value
            self.c.features[i].key = value
            self.c.features[i].value = value

    def fill_atoms(self, atom_t value, int nr_atom):
        for i in range(self.c.nr_atom):
            self.c.atoms[i] = value

    def fill_scores(self, weight_t value, int nr_class):
        for i in range(self.c.nr_class):
            self.c.scores[i] = value

    def fill_is_valid(self, int value, int nr_class):
        for i in range(self.c.nr_class):
            self.c.is_valid[i] = value
   
    def fill_costs(self, weight_t value, int nr_class):
        for i in range(self.c.nr_class):
            self.c.costs[i] = value

    def reset(self):
        self.fill_features(0, self.c.nr_feat)
        self.fill_atoms(0, self.c.nr_atom)
        self.fill_scores(0, self.c.nr_class)
        self.fill_costs(0, self.c.nr_class)
        self.fill_is_valid(1, self.c.nr_class)
   
    property features:
        def __get__(self):
            for i in range(self.nr_feat):
                yield self.c.features[i]
        def __set__(self, features):
            cdef weight_t value
            cdef int slot
            if isinstance(features, dict):
                feats_dict = features
                features = []
                for key, value in feats_dict.items():
                    if isinstance(key, int):
                        slot = 0
                    else:
                        slot, key = key
                    features.append((slot, key, value))
            self.nr_feat = len(features)
            cdef feat_t feat
            for i, (slot, feat, value) in enumerate(features):
                self.c.features[i] = FeatureC(i=slot, key=feat, value=value)

    property scores:
        def __get__(self):
            return [self.c.scores[i] for i in range(self.c.nr_class)]
        def __set__(self, scores):
            if len(scores) < self.nr_class:
                self.fill_scores(0, self.nr_class)
            else:
                self.nr_class = len(scores)
            for i, score in enumerate(scores):
                self.c.scores[i] = score

    property is_valid:
        def __get__(self):
            return [self.c.is_valid[i] for i in range(self.c.nr_class)]
        def __set__(self, validities):
            assert len(validities) == self.c.nr_class
            for i, is_valid in enumerate(validities):
                self.c.is_valid[i] = is_valid

    property costs:
        def __get__(self):
            return [self.c.costs[i] for i in range(self.c.nr_class)]
        def __set__(self, costs):
            cdef weight_t cost
            assert len(costs) == self.c.nr_class
            for i, cost in enumerate(costs):
                self.c.costs[i] = cost

    property guess:
        def __get__(self):
            return VecVec.arg_max_if_true(
                self.c.scores, self.c.is_valid, self.c.nr_class)

    property best:
        def __get__(self):
            cdef int i
            cdef int best = -1
            # Account for negative costs, so don't use arg_max_if_zero
            for i in range(self.c.nr_class):
                if self.c.is_valid[i] \
                and self.c.costs[i] <= 0 \
                and (best == -1 or self.c.scores[i] > self.c.scores[best]):
                    best = i
            return best
    
    property cost:
        def __get__(self):
            return self.c.costs[self.guess]
        def __set__(self, weight_t value):
            self.c.costs[self.guess] = value
   
    property nr_class:
        def __get__(self):
            return self.c.nr_class
 
    property nr_atom:
        def __get__(self):
            return self.c.nr_atom
        def __set__(self, int nr_atom):
            self.resize_atoms(nr_atom)

    property nr_feat:
        def __get__(self):
            return self.c.nr_feat
        def __set__(self, int nr_feat):
            self.c.nr_feat = nr_feat
            #self.resize_features(nr_feat)

    property loss:
        def __get__(self):
            return 1 - self.c.scores[self.best]
