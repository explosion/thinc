# cython: profile=True

from libc.stdlib cimport strtoull, strtoul, atof, atoi
from libc.string cimport strtok
from libc.string cimport memcpy
from libc.string cimport memset

from murmurhash.mrmr cimport hash64
from cymem.cymem cimport Address

from preshed.maps cimport MapStruct
from preshed.maps cimport map_get

from thinc.features.extractor cimport feat_t

import random
import humanize
import cython
import ujson


DEF LINE_SIZE = 8


cdef int init_train_feat(Pool mem, TrainFeat* feat, const class_t nr_class) except -1:
    cdef class_t nr_lines = get_nr_rows(nr_class)
    feat.weights = <WeightLine**>mem.alloc(nr_lines, sizeof(WeightLine*))
    feat.meta = <MetaData**>mem.alloc(nr_lines, sizeof(MetaData*))
    feat.length = nr_lines
    assert feat.length != 0
    for i in range(feat.length):
        assert feat.weights[i] == NULL


cdef count_t get_total_count(TrainFeat* feat, const class_t n):
    cdef class_t nr_rows = get_nr_rows(n)
    cdef class_t row
    cdef class_t col

    cdef count_t total = 0
    for row in range(nr_rows):
        if feat.meta[row] == NULL:
            continue
        for col in range(LINE_SIZE):
            total += feat.meta[row][col].count
    return total


@cython.cdivision
cdef class_t get_row(const class_t clas):
    return clas / LINE_SIZE


@cython.cdivision
cdef class_t get_col(const class_t clas):
    return clas % LINE_SIZE


@cython.cdivision
cdef class_t get_nr_rows(const class_t n) except 0:
    cdef class_t nr_lines = get_row(n)
    if nr_lines == 0 or nr_lines * LINE_SIZE < n:
        nr_lines += 1
    return nr_lines


cdef int update_weight(TrainFeat* feat, const class_t clas, const weight_t inc) except -1:
    '''Update the weight for a parameter (a {feature, class} pair).'''
    cdef class_t row = get_row(clas)
    cdef class_t col = get_col(clas)
    feat.weights[row].line[col] += inc


cdef int update_accumulator(TrainFeat* feat, const class_t clas, const time_t time) except -1:
    '''Help a weight update for one (class, feature) pair for averaged models,
    e.g. Average Perceptron. Efficient averaging requires tracking the total
    weight for the feature, which requires a time-stamp so we can fast-forward
    through iterations where the weight was unchanged.'''
    cdef class_t row = get_row(clas)
    cdef class_t col = get_col(clas)
    cdef weight_t weight = feat.weights[row].line[col]
    cdef class_t unchanged = time - feat.meta[row][col].time
    feat.meta[row][col].total += unchanged * weight
    feat.meta[row][col].time = time


cdef int update_count(TrainFeat* feat, const class_t clas, const count_t inc) except -1:
    '''Help a weight update for one (class, feature) pair by tracking how often
    the feature has been updated.  Used in Adagrad and others.
    '''
    cdef class_t row = get_row(clas)
    cdef class_t col = get_col(clas)
    feat.meta[row][col].count += inc


cdef class_t gather_weights(TrainFeat* params, feat_t max_feat, class_t nr_class,
                            class_t nr_rows, WeightLine** w_lines,
                            feat_t* instance) except *:
    cdef:
        TrainFeat* feature
        feat_t feat_id
        size_t template_id
        class_t row
        
    cdef class_t i = 0
    cdef class_t f_i = 0
    while instance[i] != 0:
        if instance[i] >= max_feat:
            i += 1
            continue

        feature = &params[instance[i]]
        for row in range(feature.length):
            if feature.weights[row] == NULL:
                continue
            w_lines[f_i] = feature.weights[row]
            f_i += 1
        i += 1
    return f_i


cdef int set_scores(weight_t* scores, WeightLine** weight_lines,
                    class_t nr_rows, class_t nr_class) except -1:
    cdef:
        class_t row
        class_t col
        WeightLine* wline
        weight_t* row_scores
        class_t max_col
    for row in range(nr_rows):
        wline = weight_lines[row]
        row_scores = &scores[wline.start]
        max_col = nr_class - wline.start
        if max_col > LINE_SIZE:
            max_col = LINE_SIZE
        for col in range(max_col):
            row_scores[col] += wline.line[col]

@cython.cdivision
cdef int average_weight(TrainFeat* feat, const class_t nr_class, const time_t time) except -1:
    cdef time_t unchanged
    cdef class_t row
    cdef class_t col
    for row in range(get_nr_rows(nr_class)):
        if feat.weights == NULL or feat.weights[row] == NULL:
            continue
        for col in range(LINE_SIZE):
            unchanged = (time + 1) - feat.meta[row][col].time
            feat.meta[row][col].total += unchanged * feat.weights[row].line[col]
            feat.weights[row].line[col] = feat.meta[row][col].total


cdef class LinearModel:
    def __init__(self, nr_class, nr_templates):
        self.total = 0
        self.n_corr = 0
        self.nr_class = nr_class
        self.nr_templates = nr_templates
        self.time = 0
        self.cache = ScoresCache(nr_class)
        self.mem = Pool()
        self.max_size = 1000
        self.feats = <TrainFeat*>self.mem.alloc(self.max_size, sizeof(TrainFeat))
        cdef int feat_t
        for i in range(self.max_size):
            init_train_feat(self.mem, &self.feats[i], self.nr_class)
        self.curr_size = 0
        self.scores = <weight_t*>self.mem.alloc(self.nr_class, sizeof(weight_t))
        self._weight_lines = <WeightLine**>self.mem.alloc(nr_class * nr_templates,
                                                          sizeof(WeightLine))

    def __call__(self, list py_feats):
        cdef Address feat_mem = Address(len(py_feats) + 1, sizeof(feat_t))
        cdef feat_t* instance = <feat_t*>feat_mem.ptr
        cdef feat_t feat
        cdef size_t i = 0
        for feat in py_feats:
            if feat == 0:
                continue
            instance[i] = feat
            i += 1
        instance[i] = 0
        self.score(self.scores, instance)
        py_scores = []
        for i in range(self.nr_class):
            py_scores.append(self.scores[i])
        return py_scores

    cdef TrainFeat* get_feat(self, feat_t feat_id) except NULL:
        self.curr_size = max(feat_id + 1, self.curr_size)
        if feat_id < self.max_size:
            if self.feats[feat_id].meta == NULL:
                init_train_feat(self.mem, &self.feats[feat_id], self.nr_class)
            return &self.feats[feat_id]
        self.max_size = max(feat_id, self.max_size) * 2
        self.feats = <TrainFeat*>self.mem.realloc(self.feats,
                                                  sizeof(TrainFeat) * self.max_size)

        init_train_feat(self.mem, &self.feats[feat_id], self.nr_class)
        return &self.feats[feat_id]

    cdef int score(self, weight_t* scores, feat_t* instance) except -1:
        cdef class_t f_i = gather_weights(self.feats, self.curr_size, self.nr_class,
                                          get_nr_rows(self.nr_class),
                                          self._weight_lines, instance) 
 
        memset(scores, 0, self.nr_class * sizeof(weight_t))
        set_scores(scores, self._weight_lines, f_i, self.nr_class)

    cpdef int update(self, dict updates) except -1:
        cdef class_t row
        cdef class_t col
        cdef class_t clas
        cdef size_t template_id
        cdef feat_t feat_id
        cdef TrainFeat* feat
        cdef weight_t upd
        cdef dict features
        self.time += 1
        for clas, features in updates.items():
            row = get_row(clas)
            col = get_col(clas)
            for feat_id, upd in features.items():
                if upd == 0:
                    continue
                if feat_id == 0:
                    continue
                feat = self.get_feat(feat_id)
                if feat.meta[row] == NULL:
                    feat.meta[row] = <MetaData*>self.mem.alloc(LINE_SIZE, sizeof(MetaData))
                if feat.weights[row] == NULL:
                    feat.weights[row] = <WeightLine*>self.mem.alloc(1, sizeof(WeightLine))
                    feat.weights[row].start = clas - col
                update_accumulator(feat, clas, self.time)
                update_count(feat, clas, 1)
                update_weight(feat, clas, upd)

    def end_training(self):
        cdef size_t i
        for i in range(self.curr_size):
            average_weight(&self.feats[i], self.nr_class, self.time)

    def end_train_iter(self, iter_num, feat_thresh):
        pc = lambda a, b: '%.1f' % ((float(a) / (b + 1e-100)) * 100)
        acc = pc(self.n_corr, self.total)

        map_size = 0
        cache_str = '%s cache hit' % self.cache.utilization
        size_str = humanize.naturalsize(self.mem.size, gnu=True)
        msg = "#%d: Moves %d/%d=%s. %s. %s" % (iter_num, self.n_corr, self.total, acc,
                                               cache_str, size_str)
        self.n_corr = 0
        self.total = 0
        return msg

    def freq_ranks(self):
        # Get a list of (freq, i) tuples, where i is an index into the original
        freqs = [(get_total_count(&self.feats[i], self.nr_class), i)
                 for i in range(self.curr_size)]
        # Sort the frequencies, and map them to their descending rank
        # Include the index into the key, to ensure that we get unique keys.
        # Otherwise, when there are frequency ties, we'll have a problem.
        ranks = dict((key, i) for i, key in enumerate(reversed(sorted(freqs))))
        # Now get a list of frequency ranks aligned to the original
        return [ranks[key] for key in freqs]

    def sort_by_freqs(self):
        ranks = self.freq_ranks()
        by_rank = <TrainFeat*>self.mem.alloc(self.max_size, sizeof(TrainFeat))
        for i, rank in enumerate(ranks):
            by_rank[rank] = self.feats[i]
        self.mem.free(self.feats)
        self.feats = by_rank
        return ranks

    def dump(self, file_, class_t freq_thresh=0):
        cdef size_t i
        cdef bytes line
        cdef TrainFeat* feat
        cdef class_t nr_rows = get_nr_rows(self.nr_class)
        for i in range(self.curr_size):
            feat = &self.feats[i]
            if feat.meta == NULL:
                continue
            total_freq = get_total_count(feat, self.nr_class)
            if freq_thresh >= 1 and total_freq < freq_thresh:
                continue
            for row in range(nr_rows):
                if feat.weights == NULL or feat.weights[row] == NULL:
                    continue
                line = encode_line(feat, total_freq, row, i)
                if line:
                    file_.write(line)

    def load(self, file_, freq_thresh=0):
        cdef size_t template_id
        cdef feat_t feat_id
        cdef count_t freq
        cdef class_t nr_rows, row, start
        cdef class_t col
        cdef bytes py_line
        cdef bytes token
        cdef TrainFeat* feature
        cdef WeightLine* wline
        cdef weight_t weight
        nr_rows = get_nr_rows(self.nr_class)
        nr_feats = 0
        nr_weights = 0
        for py_line in file_:
            freq, feat_id, row, start, weights = decode_line(py_line)
            if freq_thresh >= 1 and freq < freq_thresh:
                continue
            feature = self.get_feat(feat_id)
            wline = NULL
            for i in range(feature.length):
                if feature.weights[i] != NULL and feature.weights[i].start == start:
                    wline = feature.weights[i]
                    break
            else:
                wline = <WeightLine*>self.mem.alloc(1, sizeof(WeightLine))
                wline.start = start
                feature.weights[feature.length] = wline
                feature.length += 1
            for col, weight in enumerate(weights):
                wline.line[col] = weight
                nr_weights += 1



cdef class ScoresCache:
    def __init__(self, class_t scores_size, class_t max_size=10000):
        self._cache = PreshMap()
        self._pool = Pool()
        self._arrays = <weight_t**>self._pool.alloc(max_size, sizeof(weight_t*))
        cdef class_t i
        for i in range(max_size):
            self._arrays[i] = <weight_t*>self._pool.alloc(scores_size, sizeof(weight_t))
        self._scores_if_full = <weight_t*>self._pool.alloc(scores_size, sizeof(weight_t))
        self.i = 0
        self.max_size = max_size
        self.scores_size = scores_size
        self.n_hit = 0
        self.n_total = 0

    @property
    def utilization(self):
        if self.n_total == 0:
            return '0'
        return '%.2f' % ((float(self.n_hit) / self.n_total) * 100)
        
    cdef weight_t* lookup(self, class_t size, void* kernel, bint* is_hit):
        cdef weight_t** resized
        cdef uint64_t hashed = hash64(kernel, size, 0)
        cdef weight_t* scores = <weight_t*>self._cache.get(hashed)
        self.n_total += 1
        if scores != NULL:
            self.n_hit += 1
            is_hit[0] = True
            return scores
        elif self.i == self.max_size:
            return self._scores_if_full
        else:
            scores = self._arrays[self.i]
            self.i += 1
            self._cache.set(hashed, scores)
            is_hit[0] = False
            return scores
    
    def flush(self):
        self.i = 0
        self.n_hit = 0
        self.n_total = 0
        self._cache = PreshMap(self._cache.length)


cdef bytes encode_line(TrainFeat* feat, count_t freq, class_t row, feat_t i):
    line = []
    line.append(freq)
    line.append(i)
    line.append(row)
    line.append(row * LINE_SIZE)
    seen_non_zero = False
    weights = []
    for col in range(LINE_SIZE):
        val = feat.weights[row].line[col]
        weights.append(val)
        if val != 0:
            seen_non_zero = True
    if seen_non_zero:
        line.append(weights)
        return ujson.dumps(line) + '\n'
    else:
        return b''


def decode_line(bytes py_line):
    return ujson.loads(py_line)
 
