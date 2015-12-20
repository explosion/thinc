cdef class Example:
    def __init__(self, nr_class=0, nn_shape=None, features=None, label=None,
                 costs=None, is_valid=None, mem=None):
        if mem is None:
            mem = Pool()
        self.mem = mem
        nr_class = self.infer_nr_class(nr_class, nn_shape=nn_shape, label=label,
                                       costs=costs, is_valid=is_valid)
        Example.init_class(&self.c, self.mem, nr_class) 
        
        if nn_shape is not None:
            Example.init_nn(&self.c, self.mem, nn_shape)
        if features is not None: # TODO handle sparse features
            Example.init_dense(&self.c, self.mem, features)
        if costs is None:
            costs = self.infer_costs(nr_class, costs, label)
        self.costs = costs

    @classmethod
    def infer_nr_class(cls, nr_class, nn_shape=None, label=None,
                       costs=None, is_valid=None):
        if nr_class >= 1:
            return nr_class
        elif nn_shape is not None:
            return nn_shape[-1]
        elif costs is not None and is_valid is not None:
            assert len(costs) == len(is_valid)
            return len(costs)
        elif costs is not None:
            return len(costs)
        elif is_valid is not None:
            return len(is_valid)
        elif label is not None:
            return label + 1
        else:
            return 0

    @classmethod
    def infer_costs(cls, nr_class, costs, label):
        if costs is not None:
            return costs
        elif label is not None:
            costs = [1] * nr_class
            costs[label] = 0
            return costs
        else:
            return [0] * nr_class

    @classmethod
    def infer_label(cls, nr_class, label, costs, scores):
        if label is not None:
            return label
        elif costs is not None:
            max_ = None
            best = None
            for i, score in enumerate(scores):
                if costs[i] == 0 and score > best:
                    max_ = score
                    best = i
            return best
        else:
            return None

    def wipe(self):
        cdef int i
        if self.c.is_valid is not NULL:
            for i in range(self.c.nr_class):
                self.c.is_valid[i] = 0
        if self.c.costs is not NULL:
            for i in range(self.c.nr_class):
                self.c.costs[i] = 0
        if self.c.scores is not NULL:
            for i in range(self.c.nr_class):
                self.c.scores[i] = 0
        if self.c.atoms is not NULL:
            for i in range(self.c.nr_atom):
                self.c.atoms[i] = 0

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
        return self.c.fwd_state[i][j]


cdef class Batch:
    def __init__(self, nn_shape, inputs, costs):
        assert len(inputs) == len(costs)
        self.mem = Pool()
        self.c.nr_eg = len(inputs)
        self.c.egs = <ExampleC*>self.mem.alloc(self.c.nr_eg, sizeof(ExampleC))

        cdef Example eg
        for i, (x, y) in enumerate(zip(inputs, costs)):
            eg = Example(nn_shape=nn_shape, features=x, costs=y, mem=self.mem)
            self.c.egs[i] = eg.c

    def __iter__(self):
        for i in range(self.c.nr_eg):
            yield Example.from_ptr(self.mem, &self.c.egs[i])

    property loss:
        def __get__(self):
            return sum(eg.loss for eg in self)
