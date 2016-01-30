cdef class Example:
    def __init__(self, int nr_class=0, int nr_atom=0,
            int nr_feat=0, model_shape=None, Pool mem=None):
        if mem is None:
            mem = Pool()
        self.mem = mem
        if nr_class is not None:
            self.reset_classes(nr_class)
        if nr_feat is not None:
            self.reset_features(nr_feat)
        if nr_atom is not None:
            self.reset_atoms(nr_atom)
        if model_shape is not None:
            self.reset_activations(model_shape)

    cpdef int reset_features(self, int nr_feat) except -1:
        self.c.features = <FeatureC*>zero_or_alloc(self.mem, self.c.features,
            sizeof(self.c.features[0]), self.c.nr_feat, nr_feat)
        self.c.nr_feat = nr_feat

    cpdef int reset_atoms(self, int nr_atom) except -1:
        self.c.atoms = <atom_t*>zero_or_alloc(self.mem, self.c.atoms,
            sizeof(self.c.atoms[0]), self.c.nr_atom, nr_atom)
        self.c.nr_atom = nr_atom

    cpdef int reset_classes(self, int nr_class) except -1:
        self.c.is_valid = <int*>zero_or_alloc(self.mem, self.c.is_valid,
            sizeof(self.c.is_valid[0]), self.c.nr_class, nr_class)
        for i in range(nr_class):
            self.c.is_valid[i] = 1
        self.c.costs = <weight_t*>zero_or_alloc(self.mem, self.c.costs,
            sizeof(self.c.costs[0]), self.c.nr_class, nr_class)
        self.c.scores = <weight_t*>zero_or_alloc(self.mem, self.c.scores,
            sizeof(self.c.scores[0]), self.c.nr_class, nr_class)
        self.c.nr_class = nr_class

    cpdef int reset_activations(self, widths) except -1:
        # Don't use zero_or_alloc here --- we can't just clobber the pointers
        raise NotImplementedError

    def reset(self, int nr_feat=-1, int nr_class=-1, int nr_atom=-1, widths=None):
        if nr_feat >= 1:
            self.reset_features(nr_feat)
        if nr_atom >= 1:
            self.reset_atoms(nr_atom)
        if nr_class >= 1:
            self.rest_classes(nr_class)
        if widths is not None:
            self.reset_activations(widths)
   
    def set_features(self, features):
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
        self.c.features = <FeatureC*>zero_or_alloc(self.mem, self.c.features,
                sizeof(FeatureC), self.c.nr_feat, len(features))
        self.c.nr_feat = len(features)
        cdef feat_t feat
        for i, (slot, feat, value) in enumerate(features):
            self.c.features[i] = FeatureC(i=slot, key=feat, value=value)

    def set_input(self, input_):
        if len(input_) > self.c.widths[0]:
            lengths = (len(input_), self.c.widths[0])
            raise IndexError("Cannot set %d elements to input of length %d" % lengths)
        cdef int i
        cdef float value
        for i, value in enumerate(input_):
            self.c.fwd_state[0][i] = value

    def set_label(self, label):
        if label is None:
            costs = [1] * self.c.nr_class
        elif isinstance(label, int):
            costs = [1] * self.c.nr_class
            costs[label] = 0
        else:
            costs = label

        if costs is not None:
            assert len(costs) == self.c.nr_class, '%d vs %d' % (len(costs), self.c.nr_class)
            for i, cost in enumerate(costs):
                self.c.costs[i] = cost

    property features:
        def __get__(self):
            for i in range(self.nr_feat):
                yield self.c.features[i]

    property scores:
        def __get__(self):
            return [self.c.scores[i] for i in range(self.c.nr_class)]

    property is_valid:
        def __get__(self):
            return [self.c.is_valid[i] for i in range(self.c.nr_class)]

    property costs:
        def __get__(self):
            return [self.c.costs[i] for i in range(self.c.nr_class)]
        def __set__(self, costs):
            assert len(costs) == self.nr_class, len(costs)
            for i, cost in enumerate(costs):
                self.c.costs[i] = cost

    property guess:
        def __get__(self):
            return VecVec.arg_max_if_true(self.c.scores, self.c.is_valid, self.c.nr_class)

    property best:
        def __get__(self):
            return VecVec.arg_max_if_zero(self.c.scores, self.c.costs, self.c.nr_class)
    
    property cost:
        def __get__(self):
            return self.c.costs[self.guess]
    
    property nr_class:
        def __get__(self):
            return self.c.nr_class
        def __set__(self, int value):
            self.c.nr_class = value
 
    property nr_atom:
        def __get__(self):
            return self.c.nr_atom
        def __set__(self, int value):
            self.c.nr_atom = value
 
    property nr_feat:
        def __get__(self):
            return self.c.nr_feat
        def __set__(self, int value):
            self.c.nr_feat = value

    property loss:
        def __get__(self):
            return 1 - self.c.scores[self.best]

    def activation(self, int i, int j):
        # TODO: Find a way to do this better!
        return self.c.fwd_state[i][j]

    def delta(self, int i, int j):
        # TODO: Find a way to do this better!
        return self.c.bwd_state[i][j]


cdef void* zero_or_alloc(Pool mem, void* ptr, size_t elem_size,
        int old_nr, int new_nr) except *:
    if ptr is NULL:
        ptr = mem.alloc(new_nr, elem_size)
    elif new_nr > old_nr:
        ptr = mem.realloc(ptr, new_nr * elem_size)
    memset(ptr, 0, new_nr * elem_size)
    return ptr
