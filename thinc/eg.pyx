cdef class Example:
    '''
    model_shape:
        - An int for number of classes, or
        - a tuple of ints (layer widths)
    features:
        - None, or
        - a sequence of ints, or
        - a sequence of floats
        - a dict of
          - int: float
          - (int, int): float
    '''
    def __init__(self, model_shape, mem=None):
        if mem is None:
            mem = Pool()
        self.mem = mem
        if isinstance(model_shape, int):
            model_shape = (model_shape,)

        Example.init(&self.c, self.mem,
            model_shape)

    def wipe(self, widths):
        self.c.guess = 0
        self.c.best = 0
        self.c.cost = 0
        self.c.nr_feat = 0
        cdef int i
        if self.c.features is not NULL:
            self.mem.free(self.c.features)
        for i, width in enumerate(widths):
            if self.c.fwd_state is not NULL and self.c.fwd_state[i] is not NULL:
                memset(self.c.fwd_state[i],
                    0, sizeof(self.c.fwd_state[i][0]) * width)
            if self.c.bwd_state is not NULL and self.c.bwd_state[i] is not NULL:
                memset(self.c.bwd_state[i],
                    0, sizeof(self.c.bwd_state[i][0]) * width)
        if self.c.is_valid is not NULL:
            memset(self.c.is_valid,
                1, sizeof(self.c.is_valid[0]) * self.c.nr_class)
        if self.c.costs is not NULL:
            memset(self.c.costs,
                0, sizeof(self.c.costs[0]) * self.c.nr_class)
        if self.c.scores is not NULL:
            memset(self.c.scores,
                0, sizeof(self.c.scores[0]) * self.c.nr_class)
        if self.c.atoms is not NULL:
            memset(self.c.atoms,
                0, sizeof(self.c.atoms[0]) * self.c.nr_class)

    def set_features(self, features):
        cdef weight_t value
        cdef int slot
        if isinstance(features, dict):
            feats_dict = features
            features = []
            for key, value in feats_dict.items():
                if isinstance(key, int):
                    table_id = 0
                else:
                    table_id, key = key
                features.append((table_id, key, value))
        cdef feat_t feat
        self.c.nr_feat = len(features)
        self.c.features = <FeatureC*>self.mem.alloc(self.c.nr_feat, sizeof(FeatureC))
        for i, (slot, feat, value) in enumerate(features):
            self.c.features[i] = FeatureC(i=slot, key=feat, val=value)

    def set_label(self, label):
        if label is None:
            costs = [1] * self.c.nr_class
        elif isinstance(label, int):
            costs = [1] * self.c.nr_class
            costs[label] = 0
        else:
            costs = label
        self.c.guess = 0
        self.c.best = 0
        self.c.cost = 1

        if costs is not None:
            assert len(costs) == self.c.nr_class, '%d vs %d' % (len(costs), self.c.nr_class)
            for i, cost in enumerate(costs):
                self.c.costs[i] = cost
                if cost == 0:
                    self.c.best = i

    property features:
        def __get__(self):
            for i in range(self.nr_feat):
                yield self.c.features[i]

    property scores:
        def __get__(self):
            return [self.c.scores[i] for i in range(self.c.nr_class)]

    property costs:
        def __get__(self):
            return [self.c.costs[i] for i in range(self.c.nr_class)]
        def __set__(self, costs):
            assert len(costs) == self.nr_class, len(costs)
            for i, cost in enumerate(costs):
                self.c.costs[i] = cost

    property guess:
        def __get__(self):
            return self.c.guess
        def __set__(self, int value):
            self.c.guess = value

    property best:
        def __get__(self):
            return self.c.best
        def __set__(self, int value):
            self.c.best = value
    
    property cost:
        def __get__(self):
            return self.c.cost
        def __set__(self, int value):
            self.c.cost = value
    
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
            return 1 - self.c.scores[self.c.best]

    def activation(self, int i, int j):
        # TODO: Find a way to do this better!
        return self.c.fwd_state[i*2][j]

    def delta(self, int i, int j):
        # TODO: Find a way to do this better!
        return self.c.bwd_state[i*2][j]
