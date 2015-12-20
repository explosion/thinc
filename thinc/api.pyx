from libc.string cimport memset
from cymem.cymem cimport Pool
import tempfile
from os import path

from .typedefs cimport weight_t, atom_t
from .structs cimport LayerC
from .update cimport AveragedPerceptronUpdater
from .model cimport LinearModel


# Make this symbol available
from .nn import NeuralNet


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


cdef int arg_max_if_zero(const weight_t* scores, const weight_t* costs,
                         const int n_classes) nogil:
    cdef int i
    cdef int best = 0
    cdef weight_t mode = -900000
    for i in range(n_classes):
        if costs[i] == 0 and scores[i] > mode:
            mode = scores[i]
            best = i
    return best



cdef class Learner:
    @classmethod
    def load(cls, path_or_file):
        pass

    def __init__(self):
        pass

    def __call__(self, features):
        return eg

    def train(self, minibatch):
        pass

    def save(self, path_or_file):
        pass

    property W:
        def __get__(self):
            pass

        def __set__(self, value):
    
    property bias:
        def __get__(self):
            pass

        def __set__(self, value):
            pass


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
