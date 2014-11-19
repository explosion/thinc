# cython: profile=True
from libc.stdio cimport fopen, fclose, fread, fwrite, feof, fseek
from libc.errno cimport errno
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdlib cimport calloc, free

import random
import cython
from os import path

from murmurhash.mrmr cimport hash64
from cymem.cymem cimport Address

from preshed.maps cimport MapStruct
from preshed.maps cimport map_get

from .weights cimport average_weight, arg_max, new_train_feat, get_total_count
from .weights cimport update_feature
from .weights cimport gather_weights, set_scores
from .weights cimport get_nr_rows
from .weights cimport free_feature


DEF LINE_SIZE = 8


cdef class LinearModel:
    '''A linear model for online supervised classification. Currently uses
    the Averaged Perceptron algorithm to learn weights.
    Expected use is via Cython --- the Python API is impoverished and inefficient.

    Emphasis is on efficiency for multi-class classification, where the number
    of classes is in the dozens or low hundreds.  The weights data structure
    is neither fully dense nor fully sparse. Instead, it's organized into
    small "lines", roughly corresponding to a cache line.
    '''
    def __init__(self, nr_class):
        self.total = 0
        self.n_corr = 0
        self.nr_class = nr_class
        self._max_wl = 1000
        self.time = 0
        self.cache = ScoresCache(nr_class)
        self.weights = PreshMap()
        self.mem = Pool()
        self.scores = <weight_t*>self.mem.alloc(self.nr_class, sizeof(weight_t))
        self._weight_lines = <WeightLine*>self.mem.alloc(self._max_wl,
                                sizeof(WeightLine))

    def __dealloc__(self):
        # Use 'raw' memory management, instead of cymem.Pool, for weights.
        # The memory overhead of cymem becomes significant here.
        cdef MapStruct* map_
        cdef size_t i
        map_ = self.weights.c_map
        for i in range(map_.length):
            if map_.cells[i].key == 0:
                continue
            feat = <TrainFeat*>map_.cells[i].value
            if feat != NULL:
                free_feature(feat)

    def __call__(self, list feats, list values=None):
        cdef int length = len(feats)
        if values is None:
            values = [1 for _ in feats]
        cdef Address f_addr = Address(length+1, sizeof(feat_t))
        cdef Address v_addr = Address(length+1, sizeof(weight_t))
        cdef feat_t* c_feats = <feat_t*>f_addr.ptr
        cdef weight_t* c_values = <weight_t*>v_addr.ptr
        for i in range(length):
            c_feats[i] = feats[i]
            c_values[i] = values[i]
        c_feats[i+1] = 0
        c_values[i+1] = 0
        self.score(self.scores, c_feats, c_values)
        return [self.scores[i] for i in range(self.nr_class)]

    cdef class_t score(self, weight_t* scores, feat_t* features, weight_t* values) except -1:
        cdef int i = 0
        while features[i] != 0:
            i += 1
        if i * self.nr_class >= self._max_wl:
            self._max_wl = (i * self.nr_class) * 2
            size = self._max_wl * sizeof(WeightLine)
            self._weight_lines = <WeightLine*>self.mem.realloc(self._weight_lines, size)
        # TODO: Use values!
        f_i = gather_weights(self.weights.c_map, self.nr_class, self._weight_lines,
                             features, values) 

        memset(scores, 0, self.nr_class * sizeof(weight_t))
        set_scores(scores, self._weight_lines, f_i, self.nr_class)
        return arg_max(scores, self.nr_class)

    cpdef int update(self, dict counts) except -1:
        cdef TrainFeat* feat
        cdef feat_t feat_id
        cdef weight_t upd
        cdef class_t clas
        cdef int i
        self.time += 1
        for clas, feat_counts in counts.items():
            assert clas >= 0
            for feat_id, upd in feat_counts.items():
                if upd == 0:
                    continue
                feat = <TrainFeat*>self.weights.get(feat_id)
                if feat == NULL:
                    feat = new_train_feat(self.nr_class)
                    self.weights.set(feat_id, feat)
                update_feature(feat, clas, upd, self.time)

    def end_training(self):
        cdef MapStruct* map_
        cdef size_t i
        map_ = self.weights.c_map
        for i in range(map_.length):
            if map_.cells[i].key == 0:
                continue
            feat = <TrainFeat*>map_.cells[i].value
            average_weight(feat, self.nr_class, self.time)

    def end_train_iter(self, iter_num, feat_thresh):
        pc = lambda a, b: '%.1f' % ((float(a) / (b + 1e-100)) * 100)
        acc = pc(self.n_corr, self.total)

        map_size = self.weights.mem.size
        msg = "#%d: Moves %d/%d=%s" % (iter_num, self.n_corr, self.total, acc)
        self.n_corr = 0
        self.total = 0
        return msg

    def dump(self, loc, class_t freq_thresh=0):
        cdef feat_t feat_id
        cdef class_t row
        cdef size_t i
        cdef MapStruct* weights = self.weights.c_map
        cdef _Writer writer = _Writer(loc, self.nr_class, freq_thresh)
        for i in range(weights.length):
            if weights.cells[i].key == 0:
                continue
            writer.write(weights.cells[i].key, <TrainFeat*>weights.cells[i].value)
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
        cdef count_t total_freq
        cdef class_t n_rows
        if feat == NULL:
            return 0
        total_freq = get_total_count(feat, self._nr_class)
        if total_freq == 0:
            return 0
        if self._freq_thresh >= 1 and total_freq < self._freq_thresh:
            return 0
        active_rows = []
        cdef class_t row
        for row in range(get_nr_rows(self._nr_class)):
            if feat.weights[row] == NULL:
                continue
            for col in range(LINE_SIZE):
                if feat.weights[row].line[col] != 0:
                    active_rows.append(row)
                    break
        status = fwrite(&total_freq, sizeof(total_freq), 1, self._fp)
        assert status == 1
        status = fwrite(&feat_id, sizeof(feat_id), 1, self._fp)
        assert status == 1
        n_rows = len(active_rows)
        status = fwrite(&n_rows, sizeof(n_rows), 1, self._fp)
        assert status == 1
        for row in active_rows:
            status = fwrite(feat.weights[row], sizeof(WeightLine), 1, self._fp)
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
        cdef count_t total_freq
        cdef feat_t feat_id
        cdef class_t n_rows
        cdef class_t row
        cdef size_t status
        status = fread(&total_freq, sizeof(count_t), 1, self._fp)
        if status == 0:
            return 0
        status = fread(&feat_id, sizeof(feat_t), 1, self._fp)
        assert status
        status = fread(&n_rows, sizeof(n_rows), 1, self._fp)
        assert status
        
        feat = <TrainFeat*>calloc(sizeof(TrainFeat), 1)
        feat.meta = NULL
        feat.weights = <WeightLine**>calloc(sizeof(WeightLine*), n_rows)
        feat.length = n_rows
        cdef int i
        for i in range(n_rows):
            feat.weights[i] = <WeightLine*>calloc(sizeof(WeightLine), 1)
            status = fread(feat.weights[i], sizeof(WeightLine), 1, self._fp)
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
