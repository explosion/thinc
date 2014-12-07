cimport cython
from libc.math cimport sqrt
from libc.stdlib cimport calloc, free

from preshed.maps cimport map_get


DEF LINE_SIZE = 8


@cython.cdivision
cdef inline class_t get_row(const class_t clas) nogil:
    return clas / LINE_SIZE


@cython.cdivision
cdef inline class_t get_col(const class_t clas) nogil:
    return clas % LINE_SIZE


@cython.cdivision
cdef class_t get_nr_rows(const class_t n) nogil:
    cdef class_t nr_lines = getw(n)
    if nr_lines == 0 or nr_lines * LINE_SIZE < n:
        nr_lines += 1
    return nr_lines


cdef int gather_weights(MapStruct* maps, class_t nr_class,
        WeightLine* w_lines, Feature* feats, int n_feats) except -1:
    cdef:
        TrainFeat* feature
        WeightLine* feat_weights
        feat_t feat_id
        int row
    cdef int i
    cdef int f_i = 0
    for i in range(n_feats):
        feature = <TrainFeat*>map_get(&maps[i], feats[i].key)
        if feature != NULL:
            for row in range(feature.length):
                feat_weights = feature.weights[row]
                if feat_weights != NULL:
                    w_lines[f_i] = feat_weights[0]
                    f_i += 1
    return f_i


cdef int set_scores(weight_t* scores, WeightLine* weight_lines,
        class_t nr_rows, class_t nr_class) except -1:
    cdef int row, col, max_col
    cdef WeightLine* wline
    cdef weight_t* row_scores
    for row in range(nr_rows):
        wline = &weight_lines[row]
        row_scores = &scores[wline.start]
        max_col = nr_class - wline.start
        if max_col > LINE_SIZE:
            max_col = LINE_SIZE
        for col in range(max_col):
            row_scores[col] += wline.line[col]


cdef TrainFeat* new_train_feat(const class_t nr_class) except NULL:
    cdef TrainFeat* output = <TrainFeat*>calloc(1, sizeof(TrainFeat))
    cdef class_t nr_lines = get_nr_rows(nr_class)
    output.weights = <WeightLine**>calloc(nr_lines, sizeof(WeightLine*))
    output.meta = <MetaData**>calloc(nr_lines, sizeof(MetaData*))
    output.length = nr_lines
    return output


cdef void free_feature(TrainFeat* feat) nogil:
    cdef int i
    for i in range(feat.length):
        if feat.weights != NULL and feat.weights[i] != NULL:
            free(feat.weights[i])
        if feat.meta != NULL and feat.meta[i] != NULL:
            free(feat.meta[i])
    if feat.weights != NULL:
        free(feat.weights)
    if feat.meta != NULL:
        free(feat.meta)
    free(feat)


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


@cython.overflowcheck(True)
cdef int perceptron_update_feature(TrainFeat* feat, class_t clas, weight_t upd,
                                   time_t time) except -1:
    assert upd != 0
    cdef class_t row = get_row(clas)
    cdef class_t col = get_col(clas)
    if feat.meta[row] == NULL:
        feat.meta[row] = <MetaData*>calloc(LINE_SIZE, sizeof(MetaData))
    if feat.weights[row] == NULL:
        feat.weights[row] = <WeightLine*>calloc(1, sizeof(WeightLine))
    feat.weights[row].start = clas - col
    
    cdef weight_t weight = feat.weights[row].line[col]
    cdef class_t unchanged = time - feat.meta[row][col].time
    feat.meta[row][col].total += unchanged * weight
    feat.meta[row][col].time = time
    
    feat.weights[row].line[col] += upd


#DEF RHO = 0.95
#DEF EPSILON = 1e-6
#cdef weight_t _root_mean_square(weight_t prev, weight_t new) except -1:
#    return (RHO * prev) + ((1 - RHO) * new ** 2) + EPSILON
#
#
#@cython.overflowcheck(True)
#cdef int adadelta_update_feature(TrainFeat* feat, class_t clas, weight_t upd,
#                                 time_t time) except -1:
#    cdef weight_t upd, rms_upd, rms_grad
#    cdef class_t row = get_row(clas)
#    cdef class_t col = get_col(clas)
#    rms_grad = _root_mean_square(feat.meta[row][col].rms_grad, g)
#    if feat.meta[row][col].count == 1:
#        rms_upd = EPSILON
#    else:
#        rms_upd = feat.meta[row][col].rms_upd
#    upd = (rms_upd / rms_grad) * g
#    feat.weights[row].line[col] += upd
#    feat.meta[row][col].rms_grad = rms_grad
#    feat.meta[row][col].rms_upd = _root_mean_square(feat.meta[row][col].rms_upd, upd)
