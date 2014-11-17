cimport cython
from libc.math cimport sqrt

from preshed.maps cimport map_get


DEF LINE_SIZE = 8


cdef class_t arg_max(weight_t* scores, class_t n_classes) except -1:
    cdef class_t best = 0
    cdef weight_t mode = scores[0]
    cdef int i
    for i in range(1, n_classes):
        if scores[i] > mode:
            best = i
            mode = scores[i]
    return best


cdef TrainFeat* new_train_feat(Pool mem, const class_t nr_class) except NULL:
    cdef TrainFeat* output = <TrainFeat*>mem.alloc(1, sizeof(TrainFeat))
    cdef class_t nr_lines = get_nr_rows(nr_class)
    output.weights = <WeightLine**>mem.alloc(nr_lines, sizeof(WeightLine*))
    output.meta = <MetaData**>mem.alloc(nr_lines, sizeof(MetaData*))
    output.length = nr_lines
    return output


cdef count_t get_total_count(TrainFeat* feat, const class_t n) except 0:
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
cdef inline class_t get_row(const class_t clas) nogil:
    return clas / LINE_SIZE


@cython.cdivision
cdef inline class_t get_col(const class_t clas) nogil:
    return clas % LINE_SIZE


@cython.cdivision
cdef class_t get_nr_rows(const class_t n) nogil:
    cdef class_t nr_lines = get_row(n)
    if nr_lines == 0 or nr_lines * LINE_SIZE < n:
        nr_lines += 1
    return nr_lines


cdef int update_feature(Pool mem, TrainFeat* feat, class_t clas, weight_t upd,
                        time_t time) except -1:
    assert upd != 0
    cdef class_t row = get_row(clas)
    cdef class_t col = get_col(clas)
    if feat.meta[row] == NULL:
        feat.meta[row] = <MetaData*>mem.alloc(LINE_SIZE, sizeof(MetaData))
    if feat.weights[row] == NULL:
        feat.weights[row] = <WeightLine*>mem.alloc(1, sizeof(WeightLine))
    feat.weights[row].start = clas - col
    update_accumulator(feat, clas, time)
    update_count(feat, clas, 1)
    update_gradient(feat, clas, upd)
    update_weight(feat, clas, upd)


DEF RHO = 0.95
DEF ETA = 1e-6
cdef weight_t root_mean_square(weight_t prev, weight_t new) except -1:
    return (RHO * prev) + ((1 - RHO) * new ** 2) + ETA



cdef weight_t update_gradient(TrainFeat* feat, const class_t clas, const weight_t g):
    cdef class_t row = get_row(clas)
    cdef class_t col = get_col(clas)
    feat.meta[row][col].rms_grad = root_mean_square(feat.meta[row][col].rms_grad, g)

cdef int update_weight(TrainFeat* feat, const class_t clas, const weight_t g) except -1:
    '''Update the weight for a parameter (a {feature, class} pair).'''
    cdef weight_t upd, rms_upd, rms_grad
    cdef class_t row = get_row(clas)
    cdef class_t col = get_col(clas)
    if feat.meta[row][col].count == 1:
        rms_upd = ETA
    else:
        rms_upd = feat.meta[row][col].rms_upd
    rms_grad = feat.meta[row][col].rms_grad
    upd = (rms_upd / rms_grad) * g
    feat.weights[row].line[col] += upd
    feat.meta[row][col].rms_upd = root_mean_square(feat.meta[row][col].rms_upd, upd)


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


cdef int gather_weights(MapStruct* map_, class_t nr_class,
        WeightLine** w_lines, feat_t* feats) except -1:
    cdef:
        TrainFeat* feature
        feat_t feat_id
        class_t row
        class_t col

    cdef class_t nr_rows = get_nr_rows(nr_class)
        
    cdef int i = 0
    cdef class_t f_i = 0
    while feats[i] != 0:
        feat_id = feats[i]
        i += 1
        feature = <TrainFeat*>map_get(map_, feat_id)
        if feature != NULL and feature.weights != NULL:
            for row in range(feature.length):
                if feature.weights[row] == NULL:
                    continue
                w_lines[f_i] = feature.weights[row]
                f_i += 1
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
