from libc.string cimport memset
from cymem.cymem cimport Pool
import tempfile
from os import path

from .typedefs cimport weight_t, atom_t
from .update cimport AveragedPerceptronUpdater
from .model cimport LinearModel


try:
    import copy_reg
except ImportError:
    import copyreg as copy_reg


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
    def from_feats(cls, int nr_class, feats, gold=None):
        nr_feat = len(feats)
        cdef Example self = cls(nr_class, nr_feat, nr_feat, nr_feat)
        for i, (key, value) in enumerate(feats):
            self.c.features[i].key = key
            self.c.features[i].val = value
        cdef int clas
        if gold is not None:
            for clas in range(self.c.nr_class):
                self.c.costs[clas] = 1
            self.c.costs[gold] = 0
            self.c.best = gold
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


cdef class Learner:
    def __init__(self, nr_class, extracter, model, updater):
        self.extracter = extracter
        self.model = model
        self.updater = updater
        self.nr_class = nr_class
        self.nr_atom = extracter.nr_atom
        self.nr_templ = self.extracter.nr_templ
        self.nr_embed = self.extracter.nr_embed

    def __call__(self, eg):
        self.extracter(eg)
        self.model(eg)

    cdef ExampleC allocate(self, Pool mem) except *:
        return Example.init(mem, self.nr_class, self.nr_atom,
                            self.nr_templ, self.nr_embed)

    cdef void set_prediction(self, ExampleC* eg) except *:
        memset(eg.scores, 0, eg.nr_class * sizeof(eg.scores[0]))
        self.model.set_scores(eg.scores, eg.features, eg.nr_feat)
        eg.guess = arg_max_if_true(eg.scores, eg.is_valid, eg.nr_class)
        eg.best = arg_max_if_zero(eg.scores, eg.costs, eg.nr_class)

    cdef void set_costs(self, ExampleC* eg, int gold) except *:
        if gold == -1:
            memset(eg.costs, 0, eg.nr_class * sizeof(eg.costs[0]))
        else:
            memset(eg.costs, 1, eg.nr_class * sizeof(eg.costs[0]))
            eg.costs[gold] = 0

    cdef void update(self, ExampleC* eg) except *:
        self.updater.update(eg)

    def end_training(self):
        self.updater.end_training()

    def dump(self, loc):
        self.model.dump(self.nr_class, loc)

    def load(self, loc):
        self.nr_class = self.model.load(loc)


cdef class AveragedPerceptron(Learner):
    def __init__(self, nr_class, extracter):
        model = LinearModel()
        updater = AveragedPerceptronUpdater(model.weights)
        Learner.__init__(self, nr_class, extracter, model, updater)

    def __reduce__(self):
        tmp_dir = tempfile.mkdtemp()
        model_loc = path.join(tmp_dir, 'model')
        self.model.dump(self.nr_class, model_loc)
        return (unpickle_ap, (self.__class__, self.nr_class, self.extracter, model_loc))


def unpickle_ap(cls, nr_class, extracter, model_loc):
    model = cls(nr_class, extracter)
    model.load(model_loc)
    return model


copy_reg.constructor(unpickle_ap)
