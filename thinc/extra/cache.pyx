from murmurhash.mrmr cimport hash64


cdef class ScoresCache:
    def __init__(self, class_t scores_size, class_t max_size=10000):
        self._cache = PreshMap()
        self.mem = Pool()
        self._arrays = <weight_t**>self.mem.alloc(max_size, sizeof(weight_t*))
        cdef class_t i
        for i in range(max_size):
            self._arrays[i] = <weight_t*>self.mem.alloc(scores_size, sizeof(weight_t))
        self._scores_if_full = <weight_t*>self.mem.alloc(scores_size, sizeof(weight_t))
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
