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


cdef class AveragedPerceptron:
    '''A linear model for online supervised classification.
    Expected use is via Cython --- the Python API is impoverished and inefficient.

    Emphasis is on efficiency for multi-class classification, where the number
    of classes is in the dozens or low hundreds.
    '''
    def __init__(self, templates, *args, **kwargs):
        self.time = 0
        self.weights = PreshMap()
        self.averages = PreshMap()
        self.lasso_ledger = PreshMap()
        self.mem = Pool()
        self.extracter = ConjunctionExtracter(templates)
        self.learn_rate = kwargs.get('learn_rate', 0.001)
        self.l1_penalty = kwargs.get('l1_penalty', 1e-8)
        self.momentum = kwargs.get('momentum', 0.9)

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
        self.updateC(&eg.c)
        return eg.loss

    def dump(self, loc):
        cdef Writer writer = Writer(loc, self.weights.capacity)
        cdef feat_t key
        cdef size_t feat_addr
        for i, (key, feat_addr) in enumerate(self.weights.items()):
            if feat_addr != 0:
                W = <SparseArrayC*>feat_addr
                seen_non_zero = False
                while W.key >= 0:
                    if W.val != 0:
                        seen_non_zero = True
                        break
                    W += 1
                if seen_non_zero:
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

    def apply_owed_L1(self):
        cdef size_t feat_addr
        cdef feat_t feat_id
        u = self.time * self.learn_rate * self.l1_penalty
        if u == 0:
            return
        for feat_id, feat_addr in self.averages.items():
            if feat_addr != 0:
                feat = <SparseAverageC*>feat_addr
                apply_L1(feat.curr, feat.penalties,
                    self.time * self.learn_rate * self.l1_penalty)
                #update_averages(feat, self.time)
                #l1_paid = <weight_t><size_t>self.lasso_ledger.get(feat_id)
                #l1_paid += group_lasso(feat.curr, l1_paid, u)
                #self.lasso_ledger.set(feat_id, <void*><size_t>l1_paid)

    def end_training(self):
        cdef size_t feat_addr
        self.apply_owed_L1()
        for feat_id, feat_addr in self.averages.items():
            if feat_addr != 0:
                feat = <SparseAverageC*>feat_addr
                update_averages(feat, self.time+1)
                W = feat.curr
                avg = feat.avgs
                while W.key >= 0:
                    avg_i = SparseArray.find_key(avg, W.key)
                    if W.val != 0:
                        W.val = avg.val / (self.time+1)
                    W += 1
                    avg += 1

    def resume_training(self):
        cdef feat_t feat_id
        cdef size_t feat_addr
        for i, (feat_id, feat_addr) in enumerate(self.weights.items()):
            if feat_addr == 0:
                continue
            train_feat = <SparseAverageC*>self.averages.get(feat_id)
            if train_feat == NULL:
                train_feat = <SparseAverageC*>PyMem_Malloc(sizeof(SparseAverageC))
                if train_feat is NULL:
                    msg = (feat_id)
                    raise MemoryError(
                        "Error allocating memory for feature: %s" % msg)
                weights = <const SparseArrayC*>feat_addr
                train_feat.curr  = SparseArray.clone(weights)
                train_feat.avgs = SparseArray.clone(weights)
                train_feat.times = SparseArray.clone(weights)
                self.averages.set(feat_id, train_feat)

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
    def nr_active_feat(self):
        n = 0
        cdef size_t feat_addr
        for feat_addr in self.weights.values():
            if feat_addr == 0:
                continue
            feat = <const SparseArrayC*>feat_addr
            while feat.key >= 0:
                if feat.val != 0:
                    n += 1
                    break
                feat += 1
        return n

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
                self.update_weight(feat.key, best,  -feat.value * eg.costs[guess])
                self.update_weight(feat.key, guess, feat.value * eg.costs[guess])

    cpdef int update_weight(self, feat_t feat_id, class_t clas,
            weight_t grad) except -1:
        if grad == 0:
            return 0
        if len(self.averages) == 0 and len(self.weights) != 0:
            self.resume_training()
        feat = <SparseAverageC*>self.averages.get(feat_id)
        if feat == NULL:
            feat = <SparseAverageC*>PyMem_Malloc(sizeof(SparseAverageC))
            if feat is NULL:
                msg = (feat_id, clas, grad)
                raise MemoryError("Error allocating memory for feature: %s" % msg)
            feat.curr  = SparseArray.init(clas, grad)
            feat.avgs  = SparseArray.init(clas, 0)
            feat.times = SparseArray.init(clas, <weight_t>self.time)
            self.averages.set(feat_id, feat)
            feat.mom1 = NULL
            feat.mom2 = NULL
            feat.penalties = NULL
            self.weights.set(feat_id, feat.curr)
        else:
            i = SparseArray.find_key(feat.curr, clas)
            if i < 0:
                feat.curr = SparseArray.resize(feat.curr)
                feat.avgs = SparseArray.resize(feat.avgs)
                feat.times = SparseArray.resize(feat.times)
                self.weights.set(feat_id, feat.curr)
                i = SparseArray.find_key(feat.curr, clas)
            feat.curr[i].key = clas
            feat.avgs[i].key = clas
            feat.times[i].key = clas
            # Apply the last round of updates, multiplied by the time unchanged
            feat.avgs[i].val += (self.time - feat.times[i].val) * feat.curr[i].val
            feat.curr[i].val -= grad
            feat.times[i].val = self.time

    cpdef int update_weight_ftrl(
            self, feat_t feat_id, class_t clas, weight_t grad) except -1:
        if grad == 0:
            return 0
        feat = <SparseAverageC*>self.averages.get(feat_id)
        if feat == NULL:
            feat = <SparseAverageC*>PyMem_Malloc(sizeof(SparseAverageC))
            if feat is NULL:
                msg = (feat_id, clas, grad)
                raise MemoryError("Error allocating memory for feature: %s" % msg)
            feat.curr  = SparseArray.init(clas, 0)
            feat.mom1  = SparseArray.init(clas, 0)
            feat.mom2  = SparseArray.init(clas, 0)
            feat.penalties  = SparseArray.init(clas, 0)
            feat.avgs  = SparseArray.init(clas, 0)
            feat.times = SparseArray.init(clas, 0)
            self.averages.set(feat_id, feat)
            self.weights.set(feat_id, feat.curr)
            i = 0
        else:
            i = SparseArray.find_key(feat.curr, clas)
            if i < 0:
                feat.curr = SparseArray.resize(feat.curr)
                feat.mom1 = SparseArray.resize(feat.mom1)
                feat.mom2 = SparseArray.resize(feat.mom2)
                feat.avgs = SparseArray.resize(feat.avgs)
                feat.penalties = SparseArray.resize(feat.penalties)
                feat.times = SparseArray.resize(feat.times)
                self.weights.set(feat_id, feat.curr)
                i = SparseArray.find_key(feat.curr, clas)
            feat.curr[i].key = clas
            feat.mom1[i].key = clas
            feat.mom2[i].key = clas
            feat.avgs[i].key = clas
            feat.penalties[i].key = clas
            feat.times[i].key = clas
            # Apply the last round of updates, multiplied by the time unchanged
            feat.avgs[i].val += (self.time - feat.times[i].val) * feat.curr[i].val
        adam_update(&feat.curr[i].val, &feat.mom1[i].val, &feat.mom2[i].val,
            self.time, feat.times[i].val, grad, self.learn_rate, self.momentum)
        feat.times[i].val = self.time
        # Apply cumulative L1 penalty, from here:
        # http://www.aclweb.org/anthology/P09-1054
        apply_L1(feat.curr, feat.penalties,
            self.time * self.learn_rate * self.l1_penalty)
        # Group lasso
        #l1_paid = <weight_t><size_t>self.lasso_ledger.get(feat_id)
        #l1_total = self.time * self.learn_rate * self.l1_penalty
        #l1_paid += group_lasso(feat.curr, l1_paid, l1_total)
        #self.lasso_ledger.set(feat_id, <void*><size_t>l1_paid)


cdef void adam_update(weight_t* w, weight_t* m1, weight_t* m2,
        weight_t t, weight_t last_upd, weight_t grad, weight_t learn_rate, weight_t _) nogil:
     # Calculate update with Adam
     beta1 = 0.9
     beta2 = 0.999
     eps = 1e-08

     m1[0] = (m1[0] * beta1) + ((1-beta1) * grad)
     m2[0] = (m2[0] * beta2) + ((1-beta2) * grad**2)

     # Estimate the number of updates, using time from last update
     nr_update = (t * (last_upd / t)) + 1

     m1t = m1[0] / (1-beta1**nr_update)
     m2t = m2[0] / (1-beta2**nr_update)

     w[0] -= learn_rate * m1t / (sqrt(m2t) + eps)


cdef void update_averages(SparseAverageC* feat, weight_t time) nogil:
    W = feat.curr
    avg = feat.avgs
    times = feat.times
    while W.key >= 0:
        if W.key == avg.key == times.key:
            unchanged = time - times.val
            avg.val += unchanged * W.val
            times.val = time
        W += 1
        times += 1
        avg += 1


cdef int apply_L1(SparseArrayC* W, SparseArrayC* ledger, weight_t total_penalty) nogil:
    if ledger is NULL:
        return 0
    while W.key >= 0:
        u = total_penalty
        z = W.val
        q = ledger.val
        if z > 0:
            W.val = max(0, z-(u+q))
        elif z < 0:
            W.val = min(0, z+(u-q))
        ledger.val += W.val - z
        W += 1
        ledger += 1


cdef weight_t group_lasso(SparseArrayC* weights, weight_t penalty_paid,
                          weight_t total_penalty) nogil:
    norm = 0.0
    i = 0
    while weights[i].key >= 0:
        if weights[i].val > 0:
            norm += weights[i].val
        else:
            norm -= weights[i].val
        i += 1
    # Find what we want the norm to be
    target = max(0, norm - (penalty_paid + total_penalty))
    while weights.key >= 0:
        # If weights[i].val is negative, we want to add anyway ---
        # so should all work out.
        # The ideea here is to reduce the norm of the feature
        # proportionally.
        weights.val = (weights.val/norm) * target
        weights += 1
    return target - norm
