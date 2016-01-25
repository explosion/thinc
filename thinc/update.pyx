cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc

from .api cimport Example
from .typedefs cimport time_t, feat_t, weight_t, class_t
from .structs cimport SparseAverageC
from .sparse cimport SparseArray


cdef SparseAverageC* init_feat(int clas, weight_t weight, int time) except NULL:
    feat = <SparseAverageC*>PyMem_Malloc(sizeof(SparseAverageC))
    feat.curr  = SparseArray.init(clas, weight)
    feat.avgs  = SparseArray.init(clas, 0)
    feat.times = SparseArray.init(clas, <weight_t>time)
    return feat


cdef int update_feature(SparseAverageC* feat, int key, weight_t upd, int time) except -1:
    cdef int i = SparseArray.find_key(feat.curr, key)
    if i >= 0:
        is_resized = False
    else:
        is_resized = True
        feat.curr = SparseArray.resize(feat.curr)
        feat.avgs = SparseArray.resize(feat.avgs)
        feat.times = SparseArray.resize(feat.times)
        i = SparseArray.find_key(feat.curr, key)
   
    feat.curr[i].key = key
    feat.avgs[i].key = key
    feat.avgs[i].key = key
    # Apply the last round of updates, multiplied by the time unchanged
    feat.avgs[i].val += (time - feat.times[i].val) * feat.curr[i].val
    feat.curr[i].val += upd
    feat.times[i].val = time
    return is_resized


@cython.cdivision(True)
cdef int average_weights(SparseAverageC* feat, time_t time) except -1:
    cdef time_t unchanged

    cdef int i = 0
    while feat.curr[i].key >= 0:
        unchanged = (time + 1) - <time_t>feat.times[i].val
        feat.avgs[i].val += unchanged * feat.curr[i].val
        feat.curr[i].val = feat.avgs[i].val / time
        i += 1


cdef class Updater:
    def __init__(self, PreshMap weights):
        self.time = 0
        self.train_weights = PreshMap()
        self.weights = weights
        self.mem = Pool()

    def __dealloc__(self):
        cdef size_t feat_addr
        # Use 'raw' memory management, instead of cymem.Pool, for weights.
        # The memory overhead of cymem becomes significant here.
        if self.train_weights is not None:
            for feat_addr in self.train_weights.values():
                if feat_addr != 0:
                    feat = <SparseAverageC*>feat_addr
                    PyMem_Free(feat.avgs)
                    PyMem_Free(feat.times)

    def __call__(self, Example eg):
        raise NotImplementedError

    cdef void update(self, ExampleC* eg) except *:
        raise NotImplementedError

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        feat = <SparseAverageC*>self.train_weights.get(feat_id)
        if feat == NULL:
            feat = init_feat(clas, upd, self.time)
            self.train_weights.set(feat_id, feat)
            self.weights.set(feat_id, feat.curr)
        else:  
            is_resized = update_feature(feat, clas, upd, self.time)
            if is_resized:
                self.weights.set(feat_id, feat.curr)


cdef class AveragedPerceptronUpdater(Updater):
    def __call__(self, Example eg):
        self.update(&eg.c)

    cdef void update(self, ExampleC* eg) except *:
        cdef weight_t weight, upd
        cdef feat_t feat_id
        cdef int i
        
        self.time += 1
        
        weight = eg.costs[eg.guess]
        if weight != 0:
            for i in range(eg.nr_feat):
                feat_id = eg.features[i].key
                upd = weight * eg.features[i].val
                if upd != 0:
                    self.update_weight(feat_id, eg.best, upd)
                    self.update_weight(feat_id, eg.guess, -upd)
    
    def end_training(self):
        cdef feat_id
        cdef size_t feat_addr
        for feat_id, feat_addr in self.train_weights.items():
            if feat_addr != 0:
                average_weights(<SparseAverageC*>feat_addr, self.time)
