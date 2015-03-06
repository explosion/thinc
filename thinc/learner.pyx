from libc.stdio cimport fopen, fclose, fread, fwrite, feof, fseek
from libc.errno cimport errno
from libc.string cimport memcpy
from libc.string cimport memset
from cpython.mem cimport PyMem_Malloc, PyMem_Free

import random
import cython
from os import path

from murmurhash.mrmr cimport hash64
from cymem.cymem cimport Address

from preshed.maps cimport MapStruct
from preshed.maps cimport map_get

from .weights cimport average_weight, new_train_feat
from .weights cimport perceptron_update_feature as update_feature
from .weights cimport gather_weights, set_scores
from .weights cimport get_nr_rows
from .weights cimport free_feature


include "compile_time_constants.pxi"


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
        self._max_wl = nr_class * nr_templates
        self.time = 0
        self.cache = ScoresCache(nr_class)
        self.weights = PreshMap()
        self.mem = Pool()
        self.scores = <weight_t*>self.mem.alloc(self.nr_class, sizeof(weight_t))
        self._weight_lines = <WeightLine*>self.mem.alloc(self._max_wl,
                                sizeof(WeightLine))

    def __dealloc__(self):
        cdef size_t feat_addr
        # Use 'raw' memory management, instead of cymem.Pool, for weights.
        # The memory overhead of cymem becomes significant here.
        for feat_addr in self.weights.values():
            if feat_addr != 0:
                free_feature(<TrainFeat*>feat_addr)

    def __call__(self, list features):
        cdef Address addr = Address(len(features), sizeof(Feature))
        feats = <Feature*>addr.ptr
        for i, feat_id in enumerate(features):
            feats[i].i = i
            feats[i].key = feat_id
            feats[i].value = 1
        scores = self.get_scores(feats, len(features))
        return [scores[i] for i in range(self.nr_class)]

    cpdef int update(self, dict counts) except -1:
        cdef TrainFeat* feat
        cdef feat_t feat_id
        cdef weight_t upd
        cdef class_t clas
        cdef int i
        self.time += 1
        for clas, feat_counts in counts.items():
            assert clas >= 0
            for (i, feat_id), upd in feat_counts.items():
                if upd == 0:
                    continue
                if feat_id == 0:
                    continue
                feat = <TrainFeat*>self.weights.get(feat_id)
                if feat == NULL:
                    feat = new_train_feat(clas)
                    self.weights.set(feat_id, feat)
                update_feature(feat, clas, upd, self.time, self.nr_class)

    def end_training(self):
        cdef size_t feat_addr
        for feat_addr in self.weights.values():
            if feat_addr != 0:
                average_weight(<TrainFeat*>feat_addr, self.nr_class, self.time)

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
                writer.write(key, <TrainFeat*>feat_addr)
        writer.close()

    def load(self, loc, freq_thresh=0):
        cdef feat_t feat_id
        cdef TrainFeat* feature
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

    cdef int write(self, feat_t feat_id, TrainFeat* feat) except -1:
        cdef class_t n_rows
        if feat == NULL:
            return 0
        active_rows = []
        cdef class_t row
        for row in range(feat.length):
            for col in range(LINE_SIZE):
                if feat.weights[row].line[col] != 0:
                    active_rows.append(row)
                    break
        status = fwrite(&feat_id, sizeof(feat_id), 1, self._fp)
        assert status == 1
        n_rows = len(active_rows)
        status = fwrite(&n_rows, sizeof(n_rows), 1, self._fp)
        assert status == 1
        for row in active_rows:
            status = fwrite(&feat.weights[row], sizeof(WeightLine), 1, self._fp)
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

    cdef int read(self, Pool mem, feat_t* out_id, TrainFeat** out_feat) except -1:
        cdef feat_t feat_id
        cdef class_t n_rows
        cdef class_t row
        cdef size_t status
        status = fread(&feat_id, sizeof(feat_t), 1, self._fp)
        if status == 0:
            return 0
        assert status
        status = fread(&n_rows, sizeof(n_rows), 1, self._fp)
        assert status
        feat = <TrainFeat*>PyMem_Malloc(sizeof(TrainFeat))
        feat.meta = NULL
        feat.weights = <WeightLine*>PyMem_Malloc(sizeof(WeightLine) * n_rows)
        feat.length = n_rows
        feat._resize_at = n_rows
        cdef int i
        for i in range(n_rows):
            status = fread(&feat.weights[i], sizeof(WeightLine), 1, self._fp)
            if status == 0:
                out_feat[0] = feat
                out_id[0] = feat_id
                return 0
            assert status == 1

        out_feat[0] = feat
        out_id[0] = feat_id
        if feof(self._fp):
            return 0
        else:
            return 1
