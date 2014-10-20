# cython: profile=True
import random
import humanize
import cython

from libc.stdlib cimport strtoull, strtoul, atof, atoi
from libc.string cimport strtok
from libc.string cimport memcpy
from libc.string cimport memset

from murmurhash.mrmr cimport hash64
from cymem.cymem cimport Address

from preshed.maps cimport MapStruct
from preshed.maps cimport map_get

from .weights cimport average_weight, arg_max, new_train_feat, get_total_count
from .weights cimport update_feature
from .weights cimport gather_weights, set_scores
from .weights cimport get_nr_rows

from .instance cimport Instance


DEF LINE_SIZE = 8

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

    def __call__(self, Instance inst):
        inst.clas = self.score(inst.scores, inst.feats, inst.values, inst.n_feats)
        return inst.clas

    def __iadd__(self, Instance inst):
        self.update(inst.clas, inst.feats, inst.values, inst.n_feats)
        return self

    cdef class_t score(self, weight_t* scores, feat_t* features, weight_t* values,
            int n_feats) except *:
        f_i = gather_weights(self.weights.maps, self.nr_class,
                             n_feats, self._weight_lines,
                             features, values) 
        memset(scores, 0, self.nr_class * sizeof(weight_t))
        set_scores(scores, self._weight_lines, f_i, self.nr_class)
        return arg_max(scores, self.nr_class)

    cdef int update(self, class_t clas, feat_t* feats, weight_t* values, int n) except -1:
        cdef TrainFeat* feat
        cdef int i
        self.time += 1
        for i in range(n):
            if values[i] == 0:
                continue
            feat = <TrainFeat*>self.weights.get(i, feats[i])
            if feat == NULL:
                feat = new_train_feat(self.mem, self.nr_class)
                self.weights.set(i, feats[i], feat)
            update_feature(self.mem, feat, clas, values[i], self.time)

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
