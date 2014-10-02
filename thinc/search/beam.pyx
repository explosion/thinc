from cymem.cymem cimport Pool


cdef class Beam:
    def __init__(self, class_t nr_class, class_t width, class_t state_size=0):
        self.nr_class = nr_class
        self.width = width
        self.mem = Pool()
        self.parents = <void**>self.mem.alloc(self.width, sizeof(void*))
        self.states = <void**>self.mem.alloc(self.width, sizeof(void*))
        self.size = 1
        cdef class_t i
        if state_size != 0:
            for i in range(self.width):
                self.parents[i] = self.mem.alloc(1, state_size)

    property score:
        def __get__(self):
            return self.q.top().first

    def fill_from_list(self, list scores):
        mem = Pool()
        c_scores = <weight_t**>mem.alloc(len(scores), sizeof(weight_t*))
        for i, clas_scores in enumerate(scores):
            c_scores[i] = <weight_t*>mem.alloc(len(clas_scores), sizeof(weight_t))
            for j, score in enumerate(clas_scores):
                c_scores[i][j] = score
        self.fill(c_scores)
        return self.q.top().first

    cdef int fill(self, weight_t** scores) except -1:
        """Populate the queue from a k * n matrix of scores, where k is the
        beam-width, and n is the number of classes.
        """
        cdef Candidate candidate
        cdef Entry entry
        cdef weight_t score
        while not self.q.empty():
            self.q.pop()
        for i in range(self.width):
            for j in range(self.nr_class):
                entry = Entry(scores[i][j], Candidate(i, j))
                self.q.push(entry)

    cpdef pair[size_t, size_t] pop(self) except *:
        """Pop the current top candidate from the beam, returning the parent
        and class.
        """
        if self.q.empty():
            raise StopIteration
        cdef weight_t score
        score, (parent, clas) = self.q.top()
        self.q.pop()
        return pair[size_t, size_t](parent, clas)


cdef class MaxViolation:
    def __init__(self):
        self.delta = -1
        self.n = 0
        self.cost = 0
        self.pred = NULL
        self.gold = NULL

    cdef weight_t check(self, int cost, weight_t p_score, weight_t g_score,
                         void* p, void* g, class_t n) except -1:
        cdef weight_t d = (p_score + 1) - g_score
        if cost >= 1 and d > self.delta:
            self.cost = cost
            self.delta = d
            self.pred = p
            self.gold = g
            self.n = n
            return d
        else:
            return 0
