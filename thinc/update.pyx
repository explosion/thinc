cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc

from .api cimport Example
from .typedefs cimport time_t, feat_t, weight_t, class_t
cimport sparse


cdef TrainFeat* init_feat(int clas, weight_t weight, int time) except NULL:
    feat = <TrainFeat*>PyMem_Malloc(sizeof(TrainFeat))
    feat.curr  = sparse.init(clas, weight)
    feat.avgs  = sparse.init(clas, 0)
    feat.times = sparse.init(clas, <weight_t>time)
    return feat


cdef int update_feature(TrainFeat* feat, int key, weight_t upd, int time) except -1:
    cdef int i = sparse.find_key(feat.curr, key)
    if i >= 0:
        is_resized = False
    else:
        is_resized = True
        feat.curr = sparse.resize(feat.curr)
        feat.avgs = sparse.resize(feat.avgs)
        feat.times = sparse.resize(feat.times)
        i = sparse.find_key(feat.curr, key)
   
    feat.curr[i].key = key
    feat.avgs[i].key = key
    feat.avgs[i].key = key
    # Apply the last round of updates, multiplied by the time unchanged
    feat.avgs[i].val += (time - feat.times[i].val) * feat.curr[i].val
    feat.curr[i].val += upd
    feat.times[i].val = time
    return is_resized


@cython.cdivision(True)
cdef int average_weights(TrainFeat* feat, time_t time) except -1:
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
                    feat = <TrainFeat*>feat_addr
                    PyMem_Free(feat.avgs)
                    PyMem_Free(feat.times)

    def __call__(self, Example eg):
        raise NotImplementedError


cdef class AveragedPerceptronUpdater(Updater):
    def __call__(self, Example eg):
        cdef weight_t weight, upd
        cdef feat_t feat_id
        cdef int i
        
        self.time += 1
        
        weight = eg.c.costs[eg.c.guess]
        if weight != 0:
            for i in range(eg.c.nr_feat):
                feat_id = eg.c.features[i].key
                upd = weight * eg.c.features[i].value
                if upd != 0:
                    self.update_weight(feat_id, eg.c.best, upd)
                    self.update_weight(feat_id, eg.c.guess, -upd)

    cdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        feat = <TrainFeat*>self.train_weights.get(feat_id)
        if feat == NULL:
            feat = init_feat(clas, upd, self.time)
            self.train_weights.set(feat_id, feat)
            self.weights.set(feat_id, feat.curr)
        else:  
            is_resized = update_feature(feat, clas, upd, self.time)
            if is_resized:
                self.weights.set(feat_id, feat.curr)

    def end_training(self):
        cdef feat_id
        cdef size_t feat_addr
        for feat_id, feat_addr in self.train_weights.items():
            if feat_addr != 0:
                average_weights(<TrainFeat*>feat_addr, self.time)
