from libc.stdio cimport fopen, fclose, fread, fwrite, feof, fseek
from libc.errno cimport errno
from libc.string cimport memcpy
from libc.string cimport memset

from libc.stdlib cimport qsort
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc

import random
import cython
from os import path

from murmurhash.mrmr cimport hash64
from cymem.cymem cimport Address

from preshed.maps cimport MapStruct
from preshed.maps cimport map_get

from .typedefs cimport feat_t

from cython.parallel import prange
cimport numpy as np
import numpy as np

from contextlib import contextmanager


cimport sparse

from sparse cimport SparseArrayC
from .api cimport Example

include "compile_time_constants.pxi"


cdef struct TrainFeat:
    SparseArrayC* curr
    SparseArrayC* avgs
    SparseArrayC* times


cdef TrainFeat* init_feat(int clas, weight_t weight, int time) except NULL:
    feat = <TrainFeat*>PyMem_Malloc(sizeof(TrainFeat))
    feat.curr  = sparse.init(clas, weight)
    feat.avgs  = sparse.init(clas, 0)
    feat.times = sparse.init(clas, <weight_t>time)
    return feat


cdef int update_feature(TrainFeat* feat, int key, weight_t upd, int time) except -1:
    cdef int i = sparse.find_key(feat.curr, key)
    if i >= 0:
        is_resized = False
    else:
        is_resized = True
        feat.curr = sparse.resize(feat.curr)
        feat.avgs = sparse.resize(feat.avgs)
        feat.times = sparse.resize(feat.times)
        i = sparse.find_key(feat.curr, key)
   
    feat.curr[i].key = key
    feat.avgs[i].key = key
    feat.avgs[i].key = key
    # Apply the last round of updates, multiplied by the time unchanged
    feat.avgs[i].val += (time - feat.times[i].val) * feat.curr[i].val
    feat.curr[i].val += upd
    feat.times[i].val = time
    return is_resized


@cython.cdivision(True)
cdef int average_weights(TrainFeat* feat, time_t time) except -1:
    cdef time_t unchanged

    cdef int i = 0
    while feat.curr[i].key >= 0:
        unchanged = (time + 1) - <time_t>feat.times[i].val
        feat.avgs[i].val += unchanged * feat.curr[i].val
        feat.curr[i].val = feat.avgs[i].val / time
        i += 1


cdef class LinearModel:
    '''A linear model for online supervised classification. Currently uses
    the Averaged Perceptron algorithm to learn weights.
    Expected use is via Cython --- the Python API is impoverished and inefficient.

    Emphasis is on efficiency for multi-class classification, where the number
    of classes is in the dozens or low hundreds.
    '''
    def __init__(self, n_classes=2):
        self.total = 0
        self.n_corr = 0
        self.n_classes = n_classes
        self.time = 0
        self.weights = PreshMap()
        self.train_weights = PreshMap()
        self.mem = Pool()
        self.is_updating = False

    def __dealloc__(self):
        cdef size_t feat_addr
        # Use 'raw' memory management, instead of cymem.Pool, for weights.
        # The memory overhead of cymem becomes significant here.
        for feat_addr in self.weights.values():
            if feat_addr != 0:
                PyMem_Free(<SparseArrayC*>feat_addr)
        for feat_addr in self.train_weights.values():
            if feat_addr != 0:
                feat = <TrainFeat*>feat_addr
                PyMem_Free(feat.avgs)
                PyMem_Free(feat.times)

    def __call__(self, Example eg):
        if self.extractor is not None:
            self.extractor(eg)
        self.set_scores(eg.c.scores, self.weights.c_map, eg.c.features, eg.c.nr_feat)
        eg.c.guess = arg_max_if_true(eg.c.scores, eg.c.is_valid, eg.c.nr_class)

    @staticmethod
    cdef void set_scores(weight_t* scores, const MapStruct* weights_table,
                         const FeatureC* feats, int nr_feat) nogil:
        # This is the main bottle-neck of spaCy --- where we spend all our time.
        # Typical sizes for the dependency parser model:
        # * weights_table: ~9 million entries
        # * n_feats: ~200
        # * scores: ~80 classes
        # 
        # I think the bottle-neck is actually reading the weights from main memory.
 
        cdef int i, j
        for i in range(nr_feat):
            feat = feats[i]
            class_weights = <const SparseArrayC*>map_get(weights_table, feat.key)
            if class_weights != NULL:
                j = 0
                while class_weights[j].key >= 0:
                    scores[class_weights[j].key] += class_weights[j].val * feat.value
                    j += 1

    def begin_update(self):
        self.time += 1
        self.is_updating = True
        yield
        self.is_updating = False

    def train(self, Example eg):
        self(eg)
        eg.c.best = arg_max_if_zero(eg.c.scores, eg.c.costs, self.n_classes)
        eg.c.cost = eg.c.costs[eg.c.guess]
        self.update(eg.c.features, eg.c.nr_feat, eg.c.guess, eg.c.best, eg.c.cost)

    cdef int update(self, const Feature* feats, int nr_feat,
                    int best, int guess, weight_t weight) except -1:
        cdef int i
        with self.begin_update():
            for i in range(nr_feat):
                self.update_weight(feats[i].key, best, weight * feats[i].val)
                self.update_weight(feats[i].key, guess, -(weight * feats[i].val))

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        if not self.is_updating:
            raise ValueError("update_weight must be called inside begin_update context")
        self.n_classes = max(self.n_classes, clas + 1)
        if upd != 0 and feat_id != 0:
            feat = <TrainFeat*>self.train_weights.get(feat_id)
            if feat == NULL:
                feat = init_feat(clas, upd, self.time)
                self.train_weights.set(feat_id, feat)
                self.weights.set(feat_id, feat.curr)
            else:  
                is_resized = update_feature(feat, clas, upd, self.time)
                if is_resized:
                    self.weights.set(feat_id, feat.curr)

    def end_training(self):
        cdef feat_id
        cdef size_t feat_addr
        for feat_id, feat_addr in self.train_weights.items():
            if feat_addr != 0:
                average_weights(<TrainFeat*>feat_addr, self.time)

    def end_train_iter(self, iter_num, feat_thresh):
        pc = lambda a, b: '%.1f' % ((float(a) / (b + 1e-100)) * 100)
        acc = pc(self.n_corr, self.total)

        map_size = self.weights.mem.size
        msg = "#%d: Moves %d/%d=%s" % (iter_num, self.n_corr, self.total, acc)
        self.n_corr = 0
        self.total = 0
        return msg

    def dump(self, loc):
        cdef:
            feat_t key
            size_t i
            size_t feat_addr

        cdef _Writer writer = _Writer(loc, self.n_classes)
        for i, (key, feat_addr) in enumerate(self.weights.items()):
            if feat_addr != 0:
                writer.write(key, <SparseArrayC*>feat_addr)
        writer.close()

    def load(self, loc):
        cdef feat_t feat_id
        cdef SparseArrayC* feature
        cdef _Reader reader = _Reader(loc)
        self.n_classes = reader._nr_class
        while reader.read(self.mem, &feat_id, &feature):
            self.weights.set(feat_id, feature)


cdef class _Writer:
    def __init__(self, object loc, n_classes):
        if path.exists(loc):
            assert not path.isdir(loc)
        cdef bytes bytes_loc = loc.encode('utf8') if type(loc) == unicode else loc
        self._fp = fopen(<char*>bytes_loc, 'wb')
        assert self._fp != NULL
        fseek(self._fp, 0, 0)
        self._nr_class = n_classes
        _write(&self._nr_class, sizeof(self._nr_class), 1, self._fp)

    def close(self):
        cdef size_t status = fclose(self._fp)
        assert status == 0

    cdef int write(self, feat_t feat_id, SparseArrayC* feat) except -1:
        if feat == NULL:
            return 0
        
        _write(&feat_id, sizeof(feat_id), 1, self._fp)
        
        cdef int i = 0
        while feat[i].key >= 0:
            i += 1
        cdef int32_t length = i
        
        _write(&length, sizeof(length), 1, self._fp)
        
        qsort(feat, length, sizeof(SparseArrayC), sparse.cmp_SparseArrayC)
        
        for i in range(length):
            _write(&feat[i].key, sizeof(feat[i].key), 1, self._fp)
            _write(&feat[i].val, sizeof(feat[i].val), 1, self._fp)


cdef int _write(void* value, size_t size, int n, FILE* fp) except -1:
    status = fwrite(value, size, 1, fp)
    assert status == 1, status


cdef class _Reader:
    def __init__(self, loc):
        assert path.exists(loc)
        assert not path.isdir(loc)
        cdef bytes bytes_loc = loc.encode('utf8') if type(loc) == unicode else loc
        self._fp = fopen(<char*>bytes_loc, 'rb')
        assert self._fp != NULL
        status = fseek(self._fp, 0, 0)
        status = fread(&self._nr_class, sizeof(self._nr_class), 1, self._fp)

    def __dealloc__(self):
        fclose(self._fp)

    cdef int read(self, Pool mem, feat_t* out_id, SparseArrayC** out_feat) except -1:
        cdef feat_t feat_id
        cdef int32_t length

        status = fread(&feat_id, sizeof(feat_t), 1, self._fp)
        if status == 0:
            return 0
        assert status

        status = fread(&length, sizeof(length), 1, self._fp)
        assert status
        
        feat = <SparseArrayC*>PyMem_Malloc((length + 1) * sizeof(SparseArrayC))
        
        cdef int i
        for i in range(length):
            status = fread(&feat[i].key, sizeof(feat[i].key), 1, self._fp)
            assert status
            status = fread(&feat[i].val, sizeof(feat[i].val), 1, self._fp)
            assert status

        # Trust We allocated correctly above
        feat[length].key = -2 # Indicates end of memory region
        feat[length].val = 0


        # Copy into the output variables
        out_feat[0] = feat
        out_id[0] = feat_id
        # Signal whether to continue reading, to the outer loop
        if feof(self._fp):
            return 0
        else:
            return 1
