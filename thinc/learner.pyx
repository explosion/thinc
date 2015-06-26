from libc.stdio cimport fopen, fclose, fread, fwrite, feof, fseek
from libc.errno cimport errno
from libc.string cimport memcpy
from libc.string cimport memset
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc

from libc.stdlib cimport qsort

import random
import cython
from os import path

from murmurhash.mrmr cimport hash64
from cymem.cymem cimport Address

from preshed.maps cimport MapStruct
from preshed.maps cimport map_get

from .typedefs cimport feat_t

from cython.parallel import prange


cimport sparse

from sparse cimport SparseArrayC

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
    of classes is in the dozens or low hundreds.  The weights data structure
    is neither fully dense nor fully sparse. Instead, it's organized into
    small "lines", roughly corresponding to a cache line.
    '''
    def __init__(self, nr_class, nr_templates):
        assert nr_templates != 0
        self.total = 0
        self.n_corr = 0
        self.nr_class = nr_class
        self.nr_templates = nr_templates
        self.time = 0
        self.cache = ScoresCache(nr_class)
        self.weights = PreshMap()
        self.train_weights = PreshMap()
        self.mem = Pool()
        self.scores = <weight_t*>self.mem.alloc(self.nr_class, sizeof(weight_t))

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

    def __call__(self, list features):
        cdef Address addr = Address(len(features), sizeof(Feature))
        feats = <Feature*>addr.ptr
        for i, feat_id in enumerate(features):
            feats[i].i = i
            feats[i].key = feat_id
            feats[i].value = 1
        memset(self.scores, 0, self.nr_class * sizeof(weight_t))
        i = 0
        j = 0
        for i in range(len(features)):
            feat = <SparseArrayC*>self.weights.get(feats[i].key)
            if feat != NULL:
                j = 0
                while feat[j].key >= 0:
                    self.scores[feat[j].key] += feat[j].val
                    j += 1
        return [self.scores[i] for i in range(self.nr_class)]

    cdef const weight_t* get_scores(self, const Feature* feats, int n_feats) nogil:
        memset(self.scores, 0, self.nr_class * sizeof(weight_t))
        self.set_scores(self.scores, feats, n_feats)
        return self.scores

    cdef int set_scores(self, weight_t* scores, const Feature* feats, int n_feats) nogil:
        cdef int i = 0
        cdef int j = 0
        cdef weight_t value = 0
        cdef const MapStruct* weights_table = self.weights.c_map
       
        with gil:
            active = <SparseArrayC**>PyMem_Malloc(n_feats * sizeof(SparseArrayC*))
        for i in prange(n_feats, nogil=True):
            active[i] = <SparseArrayC*>map_get(weights_table, feats[i].key)

        cdef const SparseArrayC* weights
        for i in range(n_feats):
            weights = active[i]
            if weights != NULL:
                value = feats[i].value
                j = 0
                while weights[j].key >= 0:
                    scores[weights[j].key] += value * weights[j].val
                    j += 1
        with gil:
            PyMem_Free(active)
        return 0

    cpdef int update(self, dict updates) except -1:
        cdef feat_t feat_id
        cdef weight_t upd
        cdef class_t clas
        cdef int i
        cdef TrainFeat* feat
        self.time += 1
        for clas, feat_updates in updates.items():
            assert clas >= 0
            for (i, feat_id), upd in feat_updates.items():
                if upd == 0:
                    continue
                if feat_id == 0:
                    continue
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

    def dump(self, loc, class_t freq_thresh=0):
        cdef:
            feat_t key
            size_t i
            size_t feat_addr

        cdef _Writer writer = _Writer(loc, self.nr_class, freq_thresh)
        for i, (key, feat_addr) in enumerate(self.weights.items()):
            if feat_addr != 0:
                writer.write(key, <SparseArrayC*>feat_addr)
        writer.close()

    def load(self, loc, freq_thresh=0):
        cdef feat_t feat_id
        cdef SparseArrayC* feature
        cdef _Reader reader = _Reader(loc, self.nr_class, freq_thresh)
        while reader.read(self.mem, &feat_id, &feature):
            self.weights.set(feat_id, feature)


cdef class _Writer:
    def __init__(self, object loc, nr_class, freq_thresh):
        if path.exists(loc):
            assert not path.isdir(loc)
        cdef bytes bytes_loc = loc.encode('utf8') if type(loc) == unicode else loc
        self._fp = fopen(<char*>bytes_loc, 'wb')
        fseek(self._fp, 0, 0)
        assert self._fp != NULL
        self._nr_class = nr_class
        self._freq_thresh = freq_thresh

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
    def __init__(self, loc, nr_class, freq_thresh):
        assert path.exists(loc)
        assert not path.isdir(loc)
        cdef bytes bytes_loc = loc.encode('utf8') if type(loc) == unicode else loc
        self._fp = fopen(<char*>bytes_loc, 'rb')
        assert self._fp != NULL
        status = fseek(self._fp, 0, 0)
        assert status == 0
        self._nr_class = nr_class
        self._freq_thresh = freq_thresh

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
