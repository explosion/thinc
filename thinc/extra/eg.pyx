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
    eg.nr_layer = len(widths)

    eg.scores = <weight_t*>mem.alloc(nr_class, sizeof(eg.scores[0]))
    eg.costs = <weight_t*>mem.alloc(nr_class, sizeof(eg.costs[0]))
    eg.atoms = <atom_t*>mem.alloc(nr_atom, sizeof(eg.atoms[0]))
    eg.features = <FeatureC*>mem.alloc(nr_feat, sizeof(eg.features[0]))
        
    eg.is_valid = <int*>mem.alloc(nr_class, sizeof(eg.is_valid[0]))
    for i in range(eg.nr_class):
        eg.is_valid[i] = 1

    eg.widths = <int*>mem.alloc(len(widths), sizeof(eg.widths[0]))
    eg.fwd_state = <weight_t**>mem.alloc(len(widths), sizeof(eg.fwd_state[0]))
    eg.bwd_state = <weight_t**>mem.alloc(len(widths), sizeof(eg.bwd_state[0]))
    for i, width in enumerate(widths):
        eg.widths[i] = width
        eg.fwd_state[i] = <weight_t*>mem.alloc(width, sizeof(eg.fwd_state[i][0]))
        eg.bwd_state[i] = <weight_t*>mem.alloc(width, sizeof(eg.bwd_state[i][0]))
    return eg
    

#cdef void free_eg(ExampleC* eg) nogil:
#    free(eg.scores)
#    free(eg.costs)
#    free(eg.atoms)
#    free(eg.features)
#    free(eg.is_valid)
#    for i in range(eg.nr_layer):
#        free(eg.fwd_state[i])
#        free(eg.bwd_state[i])
#        free(eg.fwd_state)
#        free(eg.bwd_state)
#        free(eg.widths)


cdef class Example:
    def __init__(self, int nr_class=0, int nr_atom=0, int nr_feat=0, widths=None):
        self.mem = Pool()
        self.c = init_eg(self.mem, nr_class=nr_class, nr_atom=nr_atom,
                         nr_feat=nr_feat, widths=widths)

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

    def fill_state(self, weight_t value, widths):
        for i in range(self.c.nr_layer):
            for j in range(self.c.widths[i]):
                self.c.fwd_state[i][j] = value
                self.c.bwd_state[i][j] = value
    
    def reset(self):
        self.fill_features(0, self.c.nr_feat)
        self.fill_atoms(0, self.c.nr_atom)
        self.fill_scores(0, self.c.nr_class)
        self.fill_costs(0, self.c.nr_class)
        self.fill_is_valid(1, self.c.nr_class)
        self.fill_state(0, self.widths)
   
    @cython.boundscheck(False)
    def set_input(self, weight_t[:] input_):
        cdef int length = input_.shape[0]
        if length > self.c.widths[0]:
            lengths = (len(input_), self.c.widths[0])
            raise IndexError("Cannot set %d elements to input of length %d" % lengths)
        cdef int i
        cdef weight_t value
        for i in range(length):
            self.c.fwd_state[0][i] = input_[i]

    property widths:
        def __get__(self):
            return [self.c.widths[i] for i in range(self.c.nr_layer)]

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
            return VecVec.arg_max_if_zero(
                self.c.scores, self.c.costs, self.c.nr_class)
    
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

    def activation(self, int i, int j):
        # TODO: Find a way to do this better!
        return self.c.fwd_state[i][j]

    def delta(self, int i, int j):
        # TODO: Find a way to do this better!
        return self.c.bwd_state[i][j]
