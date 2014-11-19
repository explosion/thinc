from libc.string cimport memset

from .weights cimport arg_max
from .features cimport ConjFeat


from os import path


cdef class Brain:
    def __init__(self, int n_atoms, int n_classes, list templates, load_from=None):
        self.mem = Pool()
        self._extr = Extractor(templates, [ConjFeat] * len(templates))
        self._model = LinearModel(n_classes)
        if load_from is not None and path.exists(load_from):
            self._model.load(path.join(load_from))
        self.n_classes = self._model.nr_class
        self.n_atoms = n_atoms
        self.feats = <feat_t*>self.mem.alloc(self._extr.n, sizeof(feat_t))
        self.values = <weight_t*>self.mem.alloc(self._extr.n, sizeof(weight_t))
        self.scores = <weight_t*>self.mem.alloc(self._model.nr_class, sizeof(weight_t))

        self._is_valid = <bint*>self.mem.alloc(self._model.nr_class, sizeof(bint))
        self._is_gold = <bint*>self.mem.alloc(self._model.nr_class, sizeof(bint))

    cdef void score(self, weight_t* scores, atom_t* atoms):
        self._extr.extract(self.feats, self.values, atoms, NULL)
        self._model.score(scores, self.feats, self.values)

    cdef class_t predict(self, atom_t* atoms) except 0:
        self.score(self.scores, atoms)
        return arg_max(self.scores, self.n_classes)

    cdef class_t predict_among(self, atom_t* atoms, list valid_classes) except 0:
        memset(self._is_valid, 0, sizeof(bint) * self.n_classes)
        cdef int clas
        for clas in valid_classes:
            self._is_valid[clas] = True
        self.score(self.scores, atoms)
        cdef int best_i = -1
        cdef int i
        cdef weight_t best_score = -10000
        for i in range(self.n_classes):
            if self._is_valid[i] and self._scores[i] > best_score:
                best_score = self._scores[i]
                best_i = i
        # Classes offset by 1, to reserve 0 as a missing value
        return best_i + 1

    cdef tuple learn(self, atom_t* atoms, list valid_classes, list gold_classes):
        assert valid_classes
        assert gold_classes
        self.score(self.scores, atoms)
        cdef int best_p = -1
        cdef int best_g = -1
        cdef weight_t p_score = -100000
        cdef weight_t g_score = -100000
        memset(self._is_valid, False, sizeof(bint) * self.n_classes)
        cdef int clas
        for clas in valid_classes:
            self._is_valid[clas-1] = True
        memset(self._is_gold, False, sizeof(bint) * self.n_classes)
        for clas in gold_classes:
            self._is_gold[clas-1] = True
        cdef weight_t* scores = self.scores
        for i in range(self.n_classes):
            # Classes offset by 1, to reserve 0 as a missing value
            if self._is_valid[i] and scores[i] > p_score:
                best_p = i + 1
                p_score = scores[i]
            if self._is_gold[i] and (best_g == -1 or scores[i] > g_score):
                best_g = i + 1
                g_score = scores[i]
        if best_p == best_g:
            upd = {}
        else:
            upd = {best_g: {}, best_p: {}}
            self._extr.count(upd[best_g], self.feats, 1)
            self._extr.count(upd[best_p], self.feats, -1)
        self._model.update(upd)
        return (best_p, best_g)

    property feats:
        def __get__(self):
            return [self.feats[i] for i in range(self.n_feats)]

        def __set__(self, list feats):
            for i, f in enumerate(feats):
                self.feats[i] = f

    property values:
        def __get__(self):
            return [self.values[i] for i in range(self.n_feats)]

        def __set__(self, list values):
            for i, f in enumerate(values):
                self.values[i] = f

    property scores:
        def __get__(self):
            return [self.scores[i] for i in range(self.n_class)]

        def __set__(self, list scores):
            assert len(scores) == self.n_classes
            for i, s in enumerate(scores):
                self.scores[i] = s
