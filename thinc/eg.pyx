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
    def __init__(self, model_shape, blocks_per_layer=1, mem=None):
        if mem is None:
            mem = Pool()
        self.mem = mem
        if isinstance(model_shape, int):
            model_shape = (model_shape,)
        Example.init(&self.c, self.mem,
            model_shape, blocks_per_layer)

    def wipe(self, widths):
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
                0, sizeof(self.c.atoms[0]) * self.c.nr_atom)

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
            self.c.features[i] = FeatureC(i=slot, key=feat, value=value)

    def set_input(self, input_):
        if len(input_) > self.c.widths[0]:
            lengths = (len(input_), self.c.widths[0])
            raise IndexError("Cannot set %d elements to input of length %d" % lengths)
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
