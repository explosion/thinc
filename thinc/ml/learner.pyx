from libc.stdlib cimport calloc, free


DEF LINE_SIZE = 7


cdef WeightLine* new_weight_line(const C start) except NULL:
    cdef WeightLine* line = <WeightLine*>calloc(1, sizeof(WeightLine))
    line.start = start
    return line


cdef CountLine* new_count_line(const C start) except NULL:
    cdef CountLine* line = <CountLine*>calloc(1, sizeof(CountLine))
    line.start = start
    return line


cdef WeightLine** new_weight_matrix(C nr_class):
    cdef I nr_lines = get_row(nr_class)
    return <WeightLine**>calloc(nr_lines, sizeof(WeightLine*))
 

cdef CountLine** new_count_matrix(C nr_class):
    cdef I nr_lines = get_row(nr_class)
    return <CountLine**>calloc(nr_lines, sizeof(CountLine*))
 

cdef TrainFeat* new_train_feat(const C n) except NULL:
    cdef TrainFeat* output = <TrainFeat*>calloc(1, sizeof(TrainFeat))
    output.weights = new_weight_matrix(n)
    output.totals = new_weight_matrix(n)
    output.counts = new_count_matrix(n)
    output.times = new_count_matrix(n)
    return output


cdef I get_row(const C clas):
    return clas / LINE_SIZE


cdef I get_col(const C clas):
    return clas % LINE_SIZE


cdef I get_nr_rows(const C n):
    cdef I nr_lines = get_row(n)
    if n % nr_lines != 0:
        nr_lines += 1
    return nr_lines
 

cdef int update_weight(TrainFeat* feat, const C clas, const W inc) except -1:
    '''Update the weight for a parameter (a {feature, class} pair).'''
    cdef I row = get_row(clas)
    cdef I col = get_col(clas)
    feat.weights[row].line[col] += inc


cdef int update_accumulator(TrainFeat* feat, const C clas, const I time) except -1:
    '''Help a weight update for one (class, feature) pair for averaged models,
    e.g. Average Perceptron. Efficient averaging requires tracking the total
    weight for the feature, which requires a time-stamp so we can fast-forward
    through iterations where the weight was unchanged.'''
    cdef I row = get_row(clas)
    cdef I col = get_col(clas)
    cdef W weight = feat.weights[row].line[col]
    feat.totals[row].line[col] += (now - feat.times[row].line[col]) * weight
    feat.times[row].line[col] = time


cdef int update_count(TrainFeat* feat, const C clas) except -1:
    '''Help a weight update for one (class, feature) pair by tracking how often
    the feature has been updated.  Used in Adagrad and others.
    '''
    feat.counts[get_row(clas)].line[get_col(clas)] += 1


cdef int set_scores(W* scores, WeightLine* weight_lines, I nr_rows):
    cdef:
        I col
    for row in range(nr_rows):
        for col in range(LINE_SIZE):
            scores[weight_lines[col].start + row] = weight_lines[row].line[col]


cdef class LinearModel:
    def __cinit__(self, nr_class):
        self.nr_class = nr_class
        self.weights.set_empty_key(0)
        self.train_weights.set_empty_key(0)

    def __dealloc__(self):
        for (key, weight) in self.weights:
            free_weight(weight)

        for (key, train_weight) in self.train_weights:
            free_train_weight(train_weight)

    cdef I gather_weights(self, WeightLine* w_lines, F* feat_ids, I nr_active):
        cdef:
            size_t feat_addr
            WeightLine** feature
            I i, j
        
        cdef I nr_rows = get_nr_rows(self.nr_class)
        cdef I f_i = 0
        for i in range(nr_rows):
            feat_addr = self.weights[feat_ids[i]]
            if feat_addr != 0:
                feature = <WeightLine**>feat_addr
                for row in range(nr_rows):
                    if feature[row] != NULL:
                        #memcpy(output[o], feature[j], sizeof(WeightLine))
                        w_lines[f_i] = feature[row][0]
                        f_i += 1
        return f_i

    cdef int score(self, W* scores, F* features, I nr_active) except -1:
        cdef I nr_rows = nr_active * get_nr_rows(self.nr_class)
        cdef WeightLine* weights = <WeightLine*>calloc(nr_rows, sizeof(WeightLine))
        self.gather_weights(weights, features, nr_active)
        set_scores(scores, weights, nr_rows)

    cdef int update(self, dict updates) except -1:
        cdef C clas
        cdef F feat_id
        cdef TrainFeat* feat
        cdef double upd

        for clas, features in updates.items():
            for feat_id, upd in features.items():
                feature = <TrainFeat*>self.train_weights[feat_id]
                if feature == NULL:
                    feat = self.new_feat(feature)
                update_accumulator(feat, clas, now)
                update_freq(feat, clas, 1)
                update_weight(feat, clas, upd)

    def serialize(self, loc):
        pass

    def deserialize(self, loc):
        pass
