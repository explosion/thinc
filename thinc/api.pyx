from libc.string cimport memset
from cymem.cymem cimport Pool

from .typedefs cimport weight_t, atom_t


cdef int arg_max(const weight_t* scores, const int n_classes) nogil:
    cdef int i
    cdef int best = 0
    cdef weight_t mode = scores[0]
    for i in range(1, n_classes):
        if scores[i] > mode:
            mode = scores[i]
            best = i
    return best


cdef int arg_max_if_true(const weight_t* scores, const int* is_valid,
                         const int n_classes) nogil:
    cdef int i
    cdef int best = 0
    cdef weight_t mode = -900000
    for i in range(n_classes):
        if is_valid[i] and scores[i] > mode:
            mode = scores[i]
            best = i
    return best


cdef int arg_max_if_zero(const weight_t* scores, const int* costs,
                         const int n_classes) nogil:
    cdef int i
    cdef int best = 0
    cdef weight_t mode = -900000
    for i in range(n_classes):
        if costs[i] == 0 and scores[i] > mode:
            mode = scores[i]
            best = i
    return best



cdef class Example:
    @classmethod
    def from_feats(cls, int nr_class, feats):
        nr_feat = len(feats)
        cdef Example self = cls(nr_class, nr_feat, nr_feat, nr_feat)
        for i, (key, value) in enumerate(feats):
            self.c.features[i].key = key
            self.c.features[i].value = value
        return self

    def __init__(self, int nr_class, int nr_atom, int nr_feat, int nr_embed):
        self.mem = Pool()
        self.c = Example.init(self.mem, nr_class, nr_atom, nr_feat, nr_embed)
        self.is_valid = <int[:nr_class]>self.c.is_valid
        self.costs = <int[:nr_class]>self.c.costs
        self.atoms = <atom_t[:nr_atom]>self.c.atoms
        self.embeddings = <weight_t[:nr_embed]>self.c.embeddings
        self.scores = <weight_t[:nr_class]>self.c.scores

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
 
    property nr_embed:
        def __get__(self):
            return self.c.nr_embed
        def __set__(self, int value):
            self.c.nr_embed = value

    def wipe(self):
        cdef int i
        for i in range(self.c.nr_class):
            self.c.is_valid[i] = 0
            self.c.costs[i] = 0
            self.c.scores[i] = 0
        for i in range(self.c.nr_atom):
            self.c.atoms[i] = 0
        for i in range(self.c.nr_feat):
            self.c.embeddings[i] = 0


cdef class AveragedPerceptron:
    def __init__(self, nr_class, extracter):
        self.extracter = extracter
        self.model = LinearModel(nr_class)
        self.updater = AveragedPerceptronUpdater(nr_class, self.model.weights)
        self.nr_class = nr_class
        self.nr_atoms = extracter.nr_atom
        self.nr_templ = self.extracter.nr_templ
        self.nr_embed = self.extracter.nr_embed

    cdef ExampleC allocate(self, Pool mem) except *:
        return Example.init(mem, self.nr_class, self.nr_atom,
                            self.nr_templ, self.nr_embed)

    cdef void set_prediction(self, ExampleC* eg) except *:
        memset(eg.scores, 0, eg.nr_class * sizeof(eg.scores[0]))
        self.model.set_scores(eg.scores, eg.features, eg.nr_feat)
        eg.guess = arg_max_if_true(eg.scores, eg.is_valid, eg.nr_class)

    cdef void set_costs(self, ExampleC* eg, int gold) except *:
        if gold == 0:
            memset(eg.costs, 0, eg.nr_class * sizeof(eg.costs[0]))
            eg.best = arg_max(eg.scores, eg.nr_class)
        else:
            memset(eg.costs, 1, eg.nr_class * sizeof(eg.costs[0]))
            eg.costs[gold] = 0
            eg.best = gold

    cdef void update(self, ExampleC* eg) except *:
        self.updater.update(eg)
