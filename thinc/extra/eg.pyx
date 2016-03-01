cdef class Example:
    def __init__(self, int nr_class=0, int nr_atom=0,
            int nr_feat=0, widths=None, Pool mem=None):
        if mem is None:
            mem = Pool()
        self.mem = mem
        self.fill_features(0, nr_feat)
        self.fill_atoms(0, nr_atom)
        self.fill_is_valid(1, nr_class)
        self.fill_scores(0, nr_class)
        self.fill_costs(0, nr_class)
        if widths is not None:
            self.c.nr_layer = len(widths)
            self.c.widths = <int*>self.mem.alloc(
                sizeof(self.c.widths[0]), len(widths))
            self.c.fwd_state = <weight_t**>self.mem.alloc(
                sizeof(self.c.fwd_state[0]), len(widths))
            self.c.bwd_state = <weight_t**>self.mem.alloc(
                sizeof(self.c.bwd_state[0]), len(widths))
            for i, width in enumerate(widths):
                self.c.widths[i] = width
                self.c.fwd_state[i] = <weight_t*>self.mem.alloc(
                    sizeof(self.c.fwd_state[i][0]), width)
                self.c.bwd_state[i] = <weight_t*>self.mem.alloc(
                    sizeof(self.c.bwd_state[i][0]), width)

    cpdef int fill_features(self, int value, int nr_feat) except -1:
        if self.c.features == NULL:
            self.c.features = <FeatureC*>self.mem.alloc(
                sizeof(self.c.features[0]), nr_feat)
            self.c.nr_feat = nr_feat
        elif self.c.nr_feat < nr_feat:
            self.c.features = <FeatureC*>self.mem.realloc(self.c.features,
                sizeof(self.c.features[0]) * nr_feat)
        for i in range(nr_feat):
            self.c.features[i].i = value
            self.c.features[i].key = value
            self.c.features[i].value = value
        self.c.nr_feat = nr_feat

    cpdef int fill_atoms(self, atom_t value, int nr_atom) except -1:
        if self.c.atoms == NULL:
            self.c.atoms = <atom_t*>self.mem.alloc(sizeof(self.c.atoms[0]), nr_atom)
            self.c.nr_atom = nr_atom
        elif self.c.nr_atom < nr_atom:
            self.c.atoms = <atom_t*>self.mem.realloc(self.c.atoms,
                sizeof(self.c.atoms[0]) * nr_atom)
        for i in range(nr_atom):
            self.c.atoms[i] = value
        self.c.nr_atom = nr_atom

    cpdef int fill_scores(self, weight_t value, int nr_class) except -1:
        if self.c.scores == NULL:
            self.c.scores = <weight_t*>self.mem.alloc(sizeof(self.c.scores[0]), nr_class)
            self.c.nr_class = nr_class
        elif self.c.nr_class < nr_class:
            self.c.scores = <weight_t*>self.mem.realloc(self.c.scores,
                sizeof(self.c.scores[0]) * nr_class)
        for i in range(nr_class):
            self.c.scores[i] = value
        self.c.nr_class = nr_class

    cpdef int fill_is_valid(self, int value, int nr_class) except -1:
        if self.c.is_valid == NULL:
            self.c.is_valid = <int*>self.mem.alloc(sizeof(self.c.is_valid[0]), nr_class)
            self.c.nr_class = nr_class
        elif self.c.nr_class < nr_class:
            self.c.is_valid = <int*>self.mem.realloc(self.c.is_valid,
                sizeof(self.c.is_valid[0]) * nr_class)
        for i in range(nr_class):
            self.c.is_valid[i] = value
        self.c.nr_class = nr_class
   
    cpdef int fill_costs(self, weight_t value, int nr_class) except -1:
        if self.c.costs == NULL:
            self.c.costs = <weight_t*>self.mem.alloc(sizeof(self.c.costs[0]), nr_class)
            self.c.nr_class = nr_class
        elif self.c.nr_class < nr_class:
            self.c.costs = <weight_t*>self.mem.realloc(self.c.costs,
                sizeof(self.c.costs[0]) * nr_class)
        for i in range(nr_class):
            self.c.costs[i] = value
        self.c.nr_class = nr_class
    
    def reset(self):
        self.fill_features(0, self.c.nr_feat)
        self.fill_atoms(0, self.c.nr_atom)
        self.fill_scores(0, self.c.nr_class)
        self.fill_costs(0, self.c.nr_class)
        self.fill_is_valid(1, self.c.nr_class)
        for i in range(self.c.nr_layer):
            memset(self.c.fwd_state[i],
                0, sizeof(self.c.fwd_state[i][0]) * self.c.widths[i])
            memset(self.c.bwd_state[i],
                0, sizeof(self.c.bwd_state[i][0]) * self.c.widths[i])
   
    def set_input(self, input_):
        if len(input_) > self.c.widths[0]:
            lengths = (len(input_), self.c.widths[0])
            raise IndexError("Cannot set %d elements to input of length %d" % lengths)
        cdef int i
        cdef float value
        for i, value in enumerate(input_):
            self.c.fwd_state[0][i] = value

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
            is_valid = list(scores)
            if len(scores) < self.nr_class:
                self.fill_scores(0, self.nr_class)
            else:
                self.nr_class = len(scores)
            for i, score in enumerate(scores):
                self.c.scores[i] = score

    property is_valid:
        def __get__(self):
            return [self.c.is_valid[i] for i in range(self.c.nr_class)]
        def __set__(self, is_valid):
            is_valid = list(is_valid)
            if len(is_valid) < self.nr_class:
                self.fill_is_valid(1, self.nr_class)
            else:
                self.nr_class = len(is_valid)
            for i, is_valid in enumerate(is_valid):
                self.c.is_valid[i] = is_valid

    property costs:
        def __get__(self):
            return [self.c.costs[i] for i in range(self.c.nr_class)]
        def __set__(self, costs):
            costs = list(costs)
            if len(costs) < self.nr_class:
                self.fill_costs(0, self.nr_class)
            else:
                self.nr_class = len(costs)
            cdef int i
            cdef weight_t cost
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
        def __set__(self, weight_t value):
            self.c.costs[self.guess] = value
   
    property nr_class:
        def __get__(self):
            return self.c.nr_class
        def __set__(self, int nr_class):
            if self.c.nr_class != nr_class:
                if self.c.scores is not NULL:
                    self.mem.free(self.c.scores)
                if self.c.is_valid is not NULL:
                    self.mem.free(self.c.is_valid)
                if self.c.costs is not NULL:
                    self.mem.free(self.c.costs)
                self.c.scores = <weight_t*>self.mem.alloc(
                    sizeof(self.c.scores[0]), nr_class)
                self.c.is_valid = <int*>self.mem.alloc(
                    sizeof(self.c.is_valid[0]), nr_class)
                self.c.costs = <weight_t*>self.mem.alloc(
                    sizeof(self.c.costs[0]), nr_class)
                self.c.nr_class = nr_class
 
    property nr_atom:
        def __get__(self):
            return self.c.nr_atom
        def __set__(self, int nr_atom):
            if self.c.nr_atom != nr_atom:
                if self.c.atoms is not NULL:
                    self.mem.free(self.c.atoms)
            self.c.atoms = <atom_t*>self.mem.alloc(
                sizeof(self.c.atoms[0]), nr_atom)
            self.c.nr_atom = nr_atom
 
    property nr_feat:
        def __get__(self):
            return self.c.nr_feat
        def __set__(self, int nr_feat):
            if self.c.nr_feat != nr_feat:
                if self.c.features is not NULL:
                    self.mem.free(self.c.features)
                self.c.features = <FeatureC*>self.mem.alloc(
                    sizeof(self.c.features[0]), nr_feat)
                self.c.nr_feat = nr_feat

    property loss:
        def __get__(self):
            return 1 - self.c.scores[self.best]

    def activation(self, int i, int j):
        # TODO: Find a way to do this better!
        return self.c.fwd_state[i][j]

    def delta(self, int i, int j):
        # TODO: Find a way to do this better!
        return self.c.bwd_state[i][j]
