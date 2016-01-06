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
    def __init__(self, model_shape, features=None, label=None, mem=None):
        if mem is None:
            mem = Pool()
        self.mem = mem
        if isinstance(model_shape, int):
            model_shape = (model_shape,)

        if label is None:
            costs = [1] * model_shape[-1]
        elif isinstance(label, int):
            costs = [1] * model_shape[-1]
            costs[label] = 0
        else:
            costs = label

        if features is None:
            features = []
        elif isinstance(features, dict):
            feats_dict = features
            features = []
            for key, value in feats_dict.items():
                if isinstance(key, int):
                    table_id = 0
                else:
                    table_id, key = key
                features.append((table_id, key, value))
        Example.init(&self.c, self.mem,
            model_shape, features, costs)

    def wipe(self):
        cdef int i
        if self.c.is_valid is not NULL:
            for i in range(self.c.nr_class):
                self.c.is_valid[i] = 1
        if self.c.costs is not NULL:
            for i in range(self.c.nr_class):
                self.c.costs[i] = 0
        if self.c.scores is not NULL:
            for i in range(self.c.nr_class):
                self.c.scores[i] = 0
        if self.c.atoms is not NULL:
            for i in range(self.c.nr_atom):
                self.c.atoms[i] = 0

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



cdef class Batch:
    def __init__(self, nn_shape, inputs, costs, nr_weight):
        assert len(inputs) == len(costs)
        self.mem = Pool()
        self.c.nr_eg = len(inputs)
        self.c.egs = <ExampleC*>self.mem.alloc(self.c.nr_eg, sizeof(ExampleC))

        cdef Example eg
        for i, (x, y) in enumerate(zip(inputs, costs)):
            eg = Example(nn_shape, features=x, label=y, mem=self.mem)
            self.c.egs[i] = eg.c

        self.c.nr_weight = nr_weight
        self.c.gradient = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(weight_t))

    def __iter__(self):
        for i in range(self.c.nr_eg):
            yield Example.from_ptr(self.mem, &self.c.egs[i])

    property loss:
        def __get__(self):
            return sum(eg.loss for eg in self)

