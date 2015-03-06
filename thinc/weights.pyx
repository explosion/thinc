cimport cython
from libc.math cimport sqrt
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memmove
from libc.string cimport memset

from preshed.maps cimport map_get

include "compile_time_constants.pxi"


@cython.cdivision(True)
cdef inline class_t get_row(const class_t clas) nogil:
    return clas / LINE_SIZE


@cython.cdivision(True)
cdef inline class_t get_col(const class_t clas) nogil:
    return clas % LINE_SIZE


@cython.cdivision(True)
cdef class_t get_nr_rows(const class_t n) nogil:
    cdef class_t nr_lines = get_row(n)
    if nr_lines == 0 or nr_lines * LINE_SIZE < n:
        nr_lines += 1
    return nr_lines


cdef int gather_weights(MapStruct* maps, const class_t nr_class,
        WeightLine* w_lines, const Feature* feats, const int n_feats) nogil:
    cdef:
        const TrainFeat* feature
        const WeightLine* feat_weights
        feat_t key
        weight_t value
        int row
    cdef int i, j
    cdef int f_i = 0
    for i in range(n_feats):
        key = feats[i].key
        value = feats[i].value
        if key == 0 or value == 0:
            continue
        feature = <TrainFeat*>map_get(maps, key)
        if feature != NULL:
            feat_weights = feature.weights
            for row in range(feature.length):
                w_lines[f_i] = feat_weights[row]
                if value != 1:
                    for j in range(LINE_SIZE):
                        w_lines[f_i].line[j] *= value
                f_i += 1
    return f_i


cdef int set_scores(weight_t* scores, const WeightLine* weight_lines,
        const class_t nr_rows, const class_t nr_class) nogil:
    cdef int row, col, max_col
    cdef const WeightLine* wline
    cdef weight_t* row_scores
    for row in range(nr_rows):
        wline = &weight_lines[row]
        row_scores = &scores[wline.start]
        max_col = nr_class - wline.start
        if max_col > LINE_SIZE:
            max_col = LINE_SIZE
        for col in range(max_col):
            row_scores[col] += wline.line[col]


cdef TrainFeat* new_train_feat(const class_t clas) except NULL:
    cdef TrainFeat* output = <TrainFeat*>PyMem_Malloc(sizeof(TrainFeat))
    output.weights = <WeightLine*>PyMem_Malloc(sizeof(WeightLine))
    memset(output.weights, 0, sizeof(WeightLine))
    output.meta = <MDLine*>PyMem_Malloc(sizeof(MDLine))
    memset(output.meta, 0, sizeof(MDLine))
    output.length = 1
    output._resize_at = 1
    output.weights[0].start = clas - get_col(clas)
    return output


cdef void free_feature(TrainFeat* feat) nogil:
    with gil:
        PyMem_Free(feat.weights)
        PyMem_Free(feat.meta)
        PyMem_Free(feat)


cdef int average_weight(TrainFeat* feat, const class_t nr_class, const time_t time) except -1:
    cdef time_t unchanged
    cdef class_t row
    cdef class_t col
    for row in range(feat.length):
        for col in range(LINE_SIZE):
            unchanged = (time + 1) - feat.meta[row].line[col].time
            feat.meta[row].line[col].total += unchanged * feat.weights[row].line[col]
            feat.weights[row].line[col] = feat.meta[row].line[col].total / time
            #if abs(feat.weights[row].line[col]) < 1:
            #    feat.weights[row].line[col] = 0


@cython.overflowcheck(True)
cdef int perceptron_update_feature(TrainFeat* feat, class_t clas, weight_t upd,
                                   time_t time, class_t nr_classes) except -1:
    assert upd != 0
    cdef class_t col = get_col(clas)
    cdef class_t start = clas - col
    cdef int i
    for i in range(feat.length):
        if feat.weights[i].start == start:
            row = i
            break
        if feat.weights[i].start > start:
            row = i
            _insert_row(feat, i, start, nr_classes)
            break
    else:
        row = feat.length
        _insert_row(feat, feat.length, start, nr_classes)
    cdef weight_t weight = feat.weights[row].line[col]
    cdef class_t unchanged = time - feat.meta[row].line[col].time
    feat.meta[row].line[col].total += unchanged * weight
    feat.meta[row].line[col].time = time
    feat.weights[row].line[col] += upd


cdef int _insert_row(TrainFeat* feat, int i, class_t start, class_t nr_classes) except -1:
    cdef class_t nr_rows = get_nr_rows(nr_classes)
    if feat.length == feat._resize_at:
        new_size = (feat.length +1) * 2 if (feat.length+1) * 2 < nr_rows else nr_rows
        feat.weights = <WeightLine*>PyMem_Realloc(feat.weights, new_size * sizeof(WeightLine))
        feat.meta = <MDLine*>PyMem_Realloc(feat.meta, new_size * sizeof(MDLine))
        feat._resize_at = new_size
    memmove(&feat.weights[i+1], &feat.weights[i], (feat.length - i) * sizeof(WeightLine))
    memmove(&feat.meta[i+1], &feat.meta[i], (feat.length - i) * sizeof(MDLine))

    memset(&feat.weights[i], 0, sizeof(WeightLine))
    memset(&feat.meta[i], 0, sizeof(MDLine))

    feat.weights[i].start = start
    feat.length += 1



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
