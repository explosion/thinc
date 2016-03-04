# cython: infer_types=True

cdef class Example:
    def __init__(self, int nr_class=0, int nr_atom=0,
            int nr_feat=0, widths=None, Pool mem=None):
        self.c = new ExampleC(
            widths=widths,
            nr_class=nr_class,
            nr_atom=nr_atom,
            nr_feat=nr_feat)

    def __dealloc__(self):
        del self.c

    def fill_features(self, int value, int nr_feat):
        self.c.fill_features(value)

    def fill_atoms(self, atom_t value, int nr_atom):
        self.c.fill_atoms(value)

    def fill_scores(self, weight_t value, int nr_class):
        self.c.fill_scores(value)

    def fill_is_valid(self, int value, int nr_class):
        self.c.fill_is_valid(value)
   
    def fill_costs(self, weight_t value, int nr_class):
        self.c.fill_costs(value)

    def fill_state(self, weight_t value, widths):
        self.c.fill_state(value)
    
    def reset(self):
        self.fill_features(0, self.c.nr_feat)
        self.fill_atoms(0, self.c.nr_atom)
        self.fill_scores(0, self.c.nr_class)
        self.fill_costs(0, self.c.nr_class)
        self.fill_is_valid(1, self.c.nr_class)
        self.fill_state(0, self.widths)
   
    def set_input(self, input_):
        if len(input_) > self.c.widths[0]:
            lengths = (len(input_), self.c.widths[0])
            raise IndexError("Cannot set %d elements to input of length %d" % lengths)
        cdef int i
        cdef weight_t value
        for i, value in enumerate(input_):
            self.c.fwd_state[0][i] = value

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
            self.c.resize_atoms(nr_atom)

    property nr_feat:
        def __get__(self):
            return self.c.nr_feat
        def __set__(self, int nr_feat):
            self.c.resize_features(nr_feat)

    property loss:
        def __get__(self):
            return 1 - self.c.scores[self.best]

    def activation(self, int i, int j):
        # TODO: Find a way to do this better!
        return self.c.fwd_state[i][j]

    def delta(self, int i, int j):
        # TODO: Find a way to do this better!
        return self.c.bwd_state[i][j]
