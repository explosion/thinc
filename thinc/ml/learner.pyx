# cython: profile=True

from libc.stdlib cimport strtoull, strtoul, atof, atoi
from libc.string cimport strtok
from libc.string cimport memcpy
from libc.string cimport memset

from murmurhash.mrmr cimport hash64
from cymem.cymem cimport Address

import random
import humanize
import cython


DEF LINE_SIZE = 7


cdef TrainFeat* new_train_feat(Pool mem, const class_t nr_class) except NULL:
    cdef TrainFeat* output = <TrainFeat*>mem.alloc(1, sizeof(TrainFeat))
    cdef class_t nr_lines = get_nr_rows(nr_class)
    output.weights = <WeightLine**>mem.alloc(nr_lines, sizeof(WeightLine*))
    output.meta = <MetaData**>mem.alloc(nr_lines, sizeof(MetaData*))
    output.length = nr_lines
    assert output.length != 0
    for i in range(output.length):
        assert output.weights[i] == NULL
    return output


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


cdef int set_scores(weight_t* scores, WeightLine** weight_lines,
                    class_t nr_rows, class_t nr_class) except -1:
    cdef:
        class_t row
        class_t col
        WeightLine* wline
        weight_t* row_scores
    memset(scores, 0, nr_class * sizeof(weight_t))
    for row in range(nr_rows):
        wline = weight_lines[row]
        row_scores = &scores[wline.start]
        if (start + LINE_SIZE) < nr_class:
            row_scores[0] += wline.line[0]
            row_scores[1] += wline.line[1]
            row_scores[2] += wline.line[2]
            row_scores[3] += wline.line[3]
            row_scores[4] += wline.line[4]
            row_scores[5] += wline.line[5]
            row_scores[6] += wline.line[6]
        else:
            for col in range(nr_class - wline.start):
                row_scores[col] += wline.line[col]

@cython.cdivision
cdef int average_weight(TrainFeat* feat, const class_t nr_class, const time_t time) except -1:
    cdef time_t unchanged
    cdef class_t row
    cdef class_t col
    for row in range(get_nr_rows(nr_class)):
        if feat.weights[row] == NULL:
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
        self.weights = PreshMapArray(nr_templates)
        self.mem = Pool()
        self.scores = <weight_t*>self.mem.alloc(self.nr_class, sizeof(weight_t))
        self._weight_lines = <WeightLine**>self.mem.alloc(nr_class * nr_templates,
                                                         sizeof(WeightLine))

    def __call__(self, list py_feats):
        cdef Address feat_mem = Address(len(py_feats), sizeof(feat_t))
        cdef feat_t* features = <feat_t*>feat_mem.ptr
        cdef feat_t feat
        for i, feat in enumerate(py_feats):
            features[i] = feat
        self.score(self.scores, features, len(py_feats))
        py_scores = []
        for i in range(self.nr_class):
            py_scores.append(self.scores[i])
        return py_scores

    cdef TrainFeat* new_feat(self, size_t template_id, feat_t feat_id) except NULL:
        cdef TrainFeat* feat = new_train_feat(self.mem, self.nr_class)
        self.weights.set(template_id, feat_id, feat)
        return feat

    cdef size_t gather_weights(self, WeightLine** w_lines, feat_t* feat_ids, size_t nr_active) except *:
        cdef:
            TrainFeat* feature
            feat_t feat_id
            size_t template_id
            class_t row
        
        cdef class_t nr_rows = get_nr_rows(self.nr_class)
        cdef size_t f_i = 0
        for template_id in range(nr_active):
            feat_id = feat_ids[template_id]
            if feat_id == 0:
                continue
            feature = <TrainFeat*>self.weights.get(template_id, feat_id)
            if feature != NULL:
                for row in range(feature.length):
                    if feature.weights[row] == NULL:
                        continue
                    w_lines[f_i] = feature.weights[row]
                    f_i += 1
        return f_i

    cdef int score(self, weight_t* scores, feat_t* features, size_t nr_active) except -1:
        cdef size_t f_i = self.gather_weights(self._weight_lines, features, nr_active)
        set_scores(scores, self._weight_lines, f_i, self.nr_class)

    cpdef int update(self, dict updates) except -1:
        cdef class_t row
        cdef class_t col
        cdef class_t clas
        cdef size_t template_id
        cdef feat_t feat_id
        cdef TrainFeat* feat
        cdef weight_t upd
        self.time += 1
        for clas, features in updates.items():
            row = get_row(clas)
            col = get_col(clas)
            for (template_id, feat_id), upd in features.items():
                if upd == 0:
                    continue
                assert feat_id != 0
                feat = <TrainFeat*>self.weights.get(template_id, feat_id)
                if feat == NULL:
                    feat = self.new_feat(template_id, feat_id)
                if feat.meta[row] == NULL:
                    feat.meta[row] = <MetaData*>self.mem.alloc(LINE_SIZE, sizeof(MetaData))
                if feat.weights[row] == NULL:
                    feat.weights[row] = <WeightLine*>self.mem.alloc(1, sizeof(WeightLine))
                    feat.weights[row].start = clas - col
 
                update_accumulator(feat, clas, self.time)
                update_count(feat, clas, 1)
                update_weight(feat, clas, upd)

    def end_training(self):
        cdef MapStruct* map_
        cdef size_t i
        for template_id in range(self.nr_templates):
            map_ = &self.weights.maps[template_id]
            for i in range(map_.length):
                if map_.cells[i].key == 0:
                    continue
                feat = <TrainFeat*>map_.cells[i].value
                average_weight(feat, self.nr_class, self.time)

    def end_train_iter(self, iter_num, feat_thresh):
        pc = lambda a, b: '%.1f' % ((float(a) / (b + 1e-100)) * 100)
        acc = pc(self.n_corr, self.total)

        map_size = 0
        for i in range(self.nr_templates):
            map_size += self.weights.maps[i].length * sizeof(Cell)
        cache_str = '%s cache hit' % self.cache.utilization
        size_str = humanize.naturalsize(self.mem.size, gnu=True)
        size_str += ', ' + humanize.naturalsize(map_size, gnu=True)
        msg = "#%d: Moves %d/%d=%s. %s. %s" % (iter_num, self.n_corr, self.total, acc,
                                               cache_str, size_str)
        self.n_corr = 0
        self.total = 0
        return msg

    def dump(self, file_, class_t freq_thresh=0):
        cdef feat_t feat_id
        cdef class_t row
        cdef size_t i
        cdef class_t nr_rows = get_nr_rows(self.nr_class)
        cdef MapStruct* weights
        for template_id in range(self.nr_templates):
            weights = &self.weights.maps[template_id]
            for i in range(weights.length):
                if weights.cells[i].key == 0:
                    continue
                feat_id = weights.cells[i].key
                feat = <TrainFeat*>weights.cells[i].value
                if feat == NULL:
                    continue
                total_freq = get_total_count(feat, self.nr_class)
                if freq_thresh >= 1 and total_freq < freq_thresh:
                    continue
                for row in range(nr_rows):
                    if feat.weights[row] == NULL:
                        continue
                    line = []
                    line.append(str(total_freq))
                    line.append(str(template_id))
                    line.append(str(feat_id))
                    line.append(str(row))
                    line.append(str(row * LINE_SIZE))
                    seen_non_zero = False
                    for col in range(LINE_SIZE):
                        val = '%d' % feat.weights[row].line[col]
                        line.append(val)
                        if val != '0':
                            seen_non_zero = True
                    if seen_non_zero:
                        file_.write('\t'.join(line))
                        file_.write('\n')

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
        nr_rows = get_nr_rows(self.nr_class)
        nr_feats = 0
        nr_weights = 0
        for py_line in file_:
            line = <char*>py_line
            token = strtok(line, '\t')
            freq = <count_t>strtoul(token, NULL, 10)
            token = strtok(NULL, '\t')
            template_id = <class_t>strtoul(token, NULL, 10)
            token = strtok(NULL, '\t')
            feat_id = strtoul(token, NULL, 10)
            token = strtok(NULL, '\t')
            row = <class_t>strtoul(token, NULL, 10)
            token = strtok(NULL, '\t')
            start = <class_t>strtoul(token, NULL, 10)
            if freq_thresh >= 1 and freq < freq_thresh:
                continue
            feature = <TrainFeat*>self.weights.get(template_id, feat_id)
            if feature == NULL:
                nr_feats += 1
                feature = new_train_feat(self.mem, self.nr_class)
                self.weights.set(template_id, feat_id, feature)
                feature.length = 0
            wline = NULL
            for i in range(feature.length):
                if feature.weights[i].start == start:
                    wline = feature.weights[i]
                    break
            else:
                wline = <WeightLine*>self.mem.alloc(1, sizeof(WeightLine))
                wline.start = start
                feature.weights[feature.length] = wline
                feature.length += 1
 
            for col in range(LINE_SIZE):
                token = strtok(NULL, '\t')
                wline.line[col] = atoi(token)
                nr_weights += 1
        print "Loading %d class... %d weights for %d features" % (self.nr_class, nr_weights, nr_feats)


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
