# cython: infer_types=True
cimport cython

from ..linalg cimport Vec, VecVec
from libc.math cimport sqrt as c_sqrt
from libc.string cimport memset, memcpy, memmove

from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get

cimport numpy as np




cdef ExampleC* init_eg(Pool mem, int nr_class, int nr_feat, int nr_atom, int is_sparse) except NULL:
    eg = <ExampleC*>mem.alloc(1, sizeof(ExampleC))
    eg.nr_class = nr_class
    eg.nr_atom = nr_atom
    eg.nr_feat = nr_feat
    eg.is_sparse = is_sparse

    eg.scores = <weight_t*>mem.alloc(nr_class, sizeof(weight_t))
    eg.costs = <weight_t*>mem.alloc(nr_class, sizeof(weight_t))
    eg.atoms = <atom_t*>mem.alloc(nr_atom, sizeof(atom_t))
    if is_sparse:
        eg.features = mem.alloc(nr_feat, sizeof(FeatureC))
    else:
        eg.features = mem.alloc(nr_feat, sizeof(weight_t))
        
    eg.is_valid = <int*>mem.alloc(nr_class, sizeof(int))
    for i in range(eg.nr_class):
        eg.is_valid[i] = 1
    return eg
    

cdef class Example:
    @classmethod
    def dense(cls, nr_class, X, y=None):
        cdef Example eg = Example(nr_class=nr_class, nr_feat=len(X), is_sparse=False)
        dense = <weight_t*>eg.c.features
        cdef weight_t value
        cdef int i
        for i, value in enumerate(X):
            dense[i] = value
        if y is not None:
            for i in range(eg.c.nr_class):
                eg.c.costs[i] = 0 if i == y else 1 
        return eg

    def __cinit__(self, int nr_class=0, int nr_atom=0, int nr_feat=0, is_sparse=True):
        self.mem = Pool()
        self.c = init_eg(self.mem, nr_class, nr_feat, nr_atom, 1 if is_sparse else 0)

    def fill_features(self, int value, int nr_feat):
        if self.c.is_sparse:
            features = <FeatureC*>self.c.features
            for i in range(nr_feat):
                features[i].i = value
                features[i].key = value
                features[i].value = value
        else:
            dense = <weight_t*>self.c.features
            for i in range(nr_feat):
                dense[i] = <weight_t>value

    def fill_atoms(self, atom_t value, int nr_atom):
        for i in range(self.c.nr_atom):
            self.c.atoms[i] = value

    def fill_scores(self, weight_t value, int nr_class):
        for i in range(self.c.nr_class):
            self.c.scores[i] = value

    def fill_is_valid(self, int value, int nr_class):
        for i in range(self.c.nr_class):
            self.c.is_valid[i] = value
   
    def fill_costs(self, weight_t value, int nr_class):
        for i in range(self.c.nr_class):
            self.c.costs[i] = value

    def reset(self):
        self.fill_features(0, self.c.nr_feat)
        self.fill_atoms(0, self.c.nr_atom)
        self.fill_scores(0, self.c.nr_class)
        self.fill_costs(0, self.c.nr_class)
        self.fill_is_valid(1, self.c.nr_class)
   
    property features:
        def __get__(self):
            if self.c.is_sparse:
                sparse = <FeatureC*>self.c.features
            else:
                dense = <weight_t*>self.c.features
            for i in range(self.nr_feat):
                if self.c.is_sparse:
                    yield sparse[i]
                else:
                    yield dense[i]
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
            if self.c.is_sparse:
                sparse = <FeatureC*>self.c.features
                for i, (slot, feat, value) in enumerate(features):
                    sparse[i] = FeatureC(i=slot, key=feat, value=value)
            else:
                dense = <weight_t*>self.c.features
                for slot, _, value in features:
                    dense[slot] = value

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
            self.resize_atoms(nr_atom)

    property nr_feat:
        def __get__(self):
            return self.c.nr_feat
        def __set__(self, int nr_feat):
            self.c.nr_feat = nr_feat
            #self.resize_features(nr_feat)

    property loss:
        def __get__(self):
            return 1 - self.c.scores[self.best]
