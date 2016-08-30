# cython: infer_types=True
# cython: cdivision=True
cimport cython
from os import path
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.exc cimport PyErr_CheckSignals
from libc.stdio cimport FILE, fopen, fclose, fread, fwrite, feof, fseek
from libc.errno cimport errno
from libc.string cimport memcpy
from libc.math cimport sqrt

from libc.stdlib cimport qsort
from libc.stdint cimport int32_t

from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap, MapStruct, map_get

from .sparse cimport SparseArray

from ..extra.eg cimport Example
from ..extra.mb cimport Minibatch
from ..structs cimport SparseArrayC, SparseAverageC
from ..typedefs cimport class_t, count_t, time_t, feat_t
from ..linalg cimport Vec, VecVec
from .serialize cimport Writer
from .serialize cimport Reader


cdef class AveragedPerceptron(Model):
    '''A linear model for online supervised classification.
    Expected use is via Cython --- the Python API is impoverished and inefficient.

    Emphasis is on efficiency for multi-class classification, where the number
    of classes is in the dozens or low hundreds.
    '''
    def __init__(self, templates, *args, **kwargs):
        self.weights = PreshMap()
        self.time = 0
        self.averages = PreshMap()
        self.mem = Pool()
        self.extracter = ConjunctionExtracter(templates)
        self.learn_rate = kwargs.get('learn_rate', 0.001)
        self.l1_penalty = kwargs.get('l1_penalty', 1e-8)
        self.momentum = kwargs.get('momentum', 0.9)
        print(self.learn_rate, self.l1_penalty, self.momentum)

    def __dealloc__(self):
        cdef size_t feat_addr
        # Use 'raw' memory management, instead of cymem.Pool, for weights.
        # The memory overhead of cymem becomes significant here.
        if self.weights is not None:
            for feat_addr in self.weights.values():
                if feat_addr != 0:
                    PyMem_Free(<SparseArrayC*>feat_addr)
        if self.averages is not None:
            for feat_addr in self.averages.values():
                if feat_addr != 0:
                    feat = <SparseAverageC*>feat_addr
                    PyMem_Free(feat.avgs)
                    PyMem_Free(feat.times)

    def __call__(self, eg_or_mb):
        cdef Example eg
        cdef Minibatch mb
        if isinstance(eg_or_mb, Example):
            eg = eg_or_mb
            self.set_scoresC(eg.c.scores, eg.c.features, eg.c.nr_feat)
        else:
            mb = eg_or_mb
            for i in range(mb.c.i):
                self.set_scoresC(mb.c.scores(i), mb.c.features(i), mb.c.nr_feat(i))
        PyErr_CheckSignals()
        return eg_or_mb

    def update(self, Example eg):
        self(eg)
        self.updateC(eg.c)
        return eg.loss

    def dump(self, loc):
        cdef Writer writer = Writer(loc, self.weights.length)
        cdef feat_t key
        cdef size_t feat_addr
        for i, (key, feat_addr) in enumerate(self.weights.items()):
            if feat_addr != 0:
                writer.write(key, <SparseArrayC*>feat_addr)
            if i % 1000 == 0:
                PyErr_CheckSignals()
        writer.close()

    def load(self, loc):
        cdef feat_t feat_id
        cdef SparseArrayC* feature
        cdef Reader reader = Reader(loc)
        self.weights = PreshMap(reader.nr_feat)
        cdef int i = 0
        while reader.read(self.mem, &feat_id, &feature):
            self.weights.set(feat_id, feature)
            if i % 1000 == 0:
                PyErr_CheckSignals()
            i += 1

    def end_training(self):
        cdef feat_id
        cdef size_t feat_addr
        cdef int i = 0
        u = self.time * self.learn_rate * self.l1_penalty
        for feat_id, feat_addr in self.averages.items():
            if feat_addr != 0:
                feat = <SparseAverageC*>feat_addr
                i = 0
                W = feat.curr
                times = feat.times
                avg = feat.avgs
                penalty = feat.penalty
                while W.key >= 0:
                    # Apply cumulative L1 penalty, from here:
                    # http://www.aclweb.org/anthology/P09-1054 
                    z = W.val
                    q = penalty.val
                    if z > 0:
                        W.val = max(0, z-(u+q))
                    elif z < 0:
                        W.val = min(0, z+(u-q))
                    # Average
                    if W.val != 0:
                        unchanged = (self.time + 1) - <time_t>times.val
                        avg.val += unchanged * W.val
                        W.val = avg.val / self.time

                    penalty.val += W.val - z
                    W += 1
                    times += 1
                    avg += 1
                    penalty += 1

    @property
    def L1(self):
        cdef long double l1 = 0.0
        cdef size_t feat_addr
        for feat_addr in self.weights.values():
            if feat_addr == 0: continue
            feat = <const SparseArrayC*>feat_addr
            i = 0
            while feat.key >= 0:
                if feat.val < 0 or feat.val > 0:
                    l1 += abs(feat.val)
                feat += 1
        return l1

    @property
    def nr_weight(self):
        n = 0
        cdef size_t feat_addr
        for feat_addr in self.weights.values():
            if feat_addr == 0:
                continue
            feat = <const SparseArrayC*>feat_addr
            while feat.key >= 0:
                if feat.val != 0:
                    n += 1
                feat += 1
        return n
    
    @property
    def nr_feat(self):
        return self.extracter.nr_templ

    cdef void set_scoresC(self, weight_t* scores,
            const FeatureC* feats, int nr_feat) nogil:
        # This is the main bottle-neck of spaCy --- where we spend all our time.
        # Typical sizes for the dependency parser model:
        # * weights_table: ~9 million entries
        # * n_feats: ~200
        # * scores: ~80 classes
        # 
        # I think the bottle-neck is actually reading the weights from main memory.
        cdef const MapStruct* weights_table = self.weights.c_map
        cdef int i
        cdef FeatureC feat
        for feat in feats[:nr_feat]:
            class_weights = <const SparseArrayC*>map_get(weights_table, feat.key)
            if class_weights != NULL:
                i = 0
                while class_weights[i].key >= 0:
                    scores[class_weights[i].key] += class_weights[i].val * feat.value
                    i += 1

    @cython.cdivision(True)
    cdef int updateC(self, const ExampleC* eg) except -1:
        self.time += 1
        guess = VecVec.arg_max_if_true(eg.scores, eg.is_valid, eg.nr_class)
        if eg.costs[guess] > 0:
            best = VecVec.arg_max_if_zero(eg.scores, eg.costs, eg.nr_class)
            for feat in eg.features[:eg.nr_feat]:
                self.update_weight(feat.key, best,   feat.value * eg.costs[guess])
                self.update_weight(feat.key, guess, -feat.value * eg.costs[guess])

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t grad) except -1:
        if grad == 0:
            return 0
        feat = <SparseAverageC*>self.averages.get(feat_id)
        if feat == NULL:
            feat = <SparseAverageC*>PyMem_Malloc(sizeof(SparseAverageC))
            if feat is NULL:
                msg = (feat_id, clas, grad)
                raise MemoryError("Error allocating memory for feature: %s" % msg)
            feat.curr  = SparseArray.init(clas, grad * -self.learn_rate)
            feat.avgs  = SparseArray.init(clas, 0)
            feat.times = SparseArray.init(clas, <weight_t>self.time)
            feat.penalty = SparseArray.init(clas, 0)
            self.averages.set(feat_id, feat)
            self.weights.set(feat_id, feat.curr)
            i = 0
        else:  
            i = SparseArray.find_key(feat.curr, clas)
            if i < 0:
                feat.curr = SparseArray.resize(feat.curr)
                feat.avgs = SparseArray.resize(feat.avgs)
                feat.times = SparseArray.resize(feat.times)
                feat.penalty = SparseArray.resize(feat.penalty)
                self.weights.set(feat_id, feat.curr)
                i = SparseArray.find_key(feat.curr, clas)
            feat.curr[i].key = clas
            feat.avgs[i].key = clas
            feat.times[i].key = clas
            feat.penalty[i].key = clas
            # Apply the last round of updates, multiplied by the time unchanged
            feat.avgs[i].val += (self.time - feat.times[i].val) * feat.curr[i].val
            # Calculate update with Adam
            feat.curr[i].val -= self.learn_rate * grad
            feat.times[i].val = self.time
        # Apply cumulative L1 penalty, from here:
        # http://www.aclweb.org/anthology/P09-1054 
        if self.l1_penalty > 0.0:
            u = self.time * self.learn_rate * self.l1_penalty
            z = feat.curr[i].val
            q = feat.penalty[i].val
            if z > 0:
                feat.curr[i].val = max(0, z-(u+q))
            elif z < 0:
                feat.curr[i].val = min(0, z+(u-q))
        feat.penalty[i].val += feat.curr[i].val - z
