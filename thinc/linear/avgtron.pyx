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
from libc.limits cimport ULLONG_MAX

import tqdm
import numpy
import numpy.random
from ..neural.util import minibatch
from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap, MapStruct, map_get

from .sparse cimport SparseArray

from ..extra.eg cimport Example
from ..extra.mb cimport Minibatch
from ..structs cimport SparseArrayC, SparseAverageC
from ..typedefs cimport atom_t, class_t, count_t, time_t, feat_t
from ..linalg cimport Vec, VecVec
from .serialize cimport Writer
from .serialize cimport Reader


cdef class AveragedPerceptron:
    '''A linear model for online supervised classification.
    Expected use is via Cython --- the Python API is impoverished and inefficient.

    Emphasis is on efficiency for multi-class classification, where the number
    of classes is in the dozens or low hundreds.
    '''
    def __init__(self, templates=None, extracter=None, *args, **kwargs):
        self.time = 0
        self.nr_out = kwargs.get('nr_out', 0)
        self.weights = PreshMap()
        self.averages = PreshMap()
        self.lasso_ledger = PreshMap()
        self.mem = Pool()
        if templates is not None:
            self.extracter = ConjunctionExtracter(templates)
        elif extracter is not None:
            self.extracter = extracter
        else:
            raise ValueError(
                "AveragedPerceptron requires one of templates or extracter")
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

    def __call__(self, atom_t[:] atoms):
        cdef Example eg = Example(
                            nr_class=self.nr_out,
                            nr_atom=len(atoms) + 2,
                            nr_feat=len(atoms) + 10)
        atoms = numpy.sort(atoms)
        memcpy(eg.c.atoms,
            &atoms[0], eg.c.nr_atom * sizeof(eg.c.atoms[0]))
        eg.c.atoms[len(atoms)] = 0
        eg.nr_feat = self.extracter.set_features(eg.c.features, eg.c.atoms)
        self.set_scoresC(eg.c.scores,
            eg.c.features, eg.c.nr_feat)
        scores = numpy.zeros((self.nr_out,), dtype='f')
        for i in range(self.nr_out):
            scores[i] = eg.c.scores[i]
        return scores

    def update(self, Example eg):
        self(eg)
        self.updateC(eg.c)
        return eg.loss

    def begin_update(self, X, weight_t dropout=0.0):
        cdef Pool mem = Pool()
        cdef int batch_size = len(X)

        cdef UpdateHandler finish_update = UpdateHandler(self, X)
        
        atoms = <atom_t**>mem.alloc(batch_size, sizeof(atom_t*))
        scores = <weight_t*>mem.alloc(batch_size * self.nr_out, sizeof(weight_t))
        for i in range(batch_size):
            atoms[i] = <atom_t*>mem.alloc(len(X[i]) + 1, sizeof(atom_t))
            for j, val in enumerate(sorted(X[i])):
                atoms[i][j] = val
            atoms[i][j+1] = 0
            finish_update.nr_feats[i] = self.extracter.set_features(
                finish_update.features[i], atoms[i])
            self.set_scoresC(&scores[i * self.nr_out],
                finish_update.features[i], finish_update.nr_feats[i])
            finish_update.predicts[i] = Vec.arg_max(
                    scores + (i*self.nr_out), self.nr_out)
        scores_ = numpy.zeros((batch_size, self.nr_out), dtype='f')
        for i in range(batch_size):
            for j in range(self.nr_out):
                scores_[i, j] = scores[i * self.nr_out + j]
        assert not numpy.isnan(scores_.sum())
        return scores_, finish_update

    def begin_training(self, train_data):
        return LinearTrainer(self, train_data)

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
    
    def apply_owed_L1(self):
        cdef size_t feat_addr
        cdef feat_t feat_id
        u = self.time * self.learn_rate * self.l1_penalty
        for feat_id, feat_addr in self.averages.items():
            if feat_addr != 0:
                feat = <SparseAverageC*>feat_addr
                update_averages(feat, self.time)
                l1_paid = <weight_t><size_t>self.lasso_ledger.get(feat_id)
                l1_paid += group_lasso(feat.curr, l1_paid, u)
                self.lasso_ledger.set(feat_id, <void*><size_t>l1_paid)
 
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
                    if W.val != 0:
                        W.val = avg.val / (self.time+1)
                    W += 1
                    avg += 1
    
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
                self.update_weight(feat.key, best,   feat.value * eg.costs[guess])
                self.update_weight(feat.key, guess, -feat.value * eg.costs[guess])

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        if upd == 0:
            return 0
        feat = <SparseAverageC*>self.averages.get(feat_id)
        if feat == NULL:
            feat = <SparseAverageC*>PyMem_Malloc(sizeof(SparseAverageC))
            if feat is NULL:
                msg = (feat_id, clas, upd)
                raise MemoryError("Error allocating memory for feature: %s" % msg)
            feat.curr  = SparseArray.init(clas, upd)
            feat.avgs  = SparseArray.init(clas, 0)
            feat.times = SparseArray.init(clas, <weight_t>self.time)
            self.averages.set(feat_id, feat)
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
            # Apply the last round of updates, multiplied by the time unchanged
            feat.avgs[i].val += (self.time - feat.times[i].val) * feat.curr[i].val
            feat.curr[i].val += upd
            feat.times[i].val = self.time


class LinearTrainer(object):
    def __init__(self, model, data):
        self.model = model
        self.optimizer = None
        self.nb_epoch = 1
        self.i = 0
        self._loss = 0.

    def __enter__(self):
        return self, self.optimizer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.end_training()

    def get_gradient(self, scores, labels):
        loss = 0.0
        grad = []
        for i, label in enumerate(labels):
            loss += (1.0-scores[i, int(label)])**2
        self._loss += loss / len(labels)
        return labels, loss

    def iterate(self, model, train_data, check_data, nb_epoch=None):
        if nb_epoch is None:
            nb_epoch = self.nb_epoch
        X, y = train_data
        for i in range(nb_epoch):
            indices = list(range(len(X)))
            numpy.random.shuffle(indices)
            for j in indices:
                yield X[j:j+1], y[j:j+1]


cdef class UpdateHandler:
    cdef Pool mem
    cdef int batch_size
    cdef int nr_class
    cdef AveragedPerceptron model
    cdef FeatureC** features
    cdef int* nr_feats
    cdef int* predicts
    def __init__(self, AveragedPerceptron model, X):
        self.mem = Pool()
        self.model = model
        self.batch_size = len(X)
        self.features = <FeatureC**>self.mem.alloc(self.batch_size, sizeof(void*))
        self.nr_feats = <int*>self.mem.alloc(self.batch_size, sizeof(int))
        self.predicts = <int*>self.mem.alloc(self.batch_size, sizeof(int))
        self.nr_class = model.nr_out
        for i, x in enumerate(X):
            self.features[i] = <FeatureC*>self.mem.alloc(
                                    len(x) + 2, sizeof(FeatureC))

    def __call__(self, labels, optimizer=None, **kwargs):
        self.model.time += 1
        for i in range(self.batch_size):
            feat_row = self.features[i]
            nr_feat = self.nr_feats[i]
            for feat in feat_row[:nr_feat]:
                if labels[i] != self.predicts[i]:
                    self.model.update_weight(feat.key, labels[i], feat.value)
                    self.model.update_weight(feat.key, self.predicts[i], -feat.value)


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
        unchanged = time - times.val
        avg.val += unchanged * W.val
        times.val = time
        W += 1
        times += 1
        avg += 1


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
