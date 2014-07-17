
cdef class Move:
    def __cinit__(self, double score, size_t clas, Move prev):
        self.clas = clas
        self.score = score
        self.prev = prev

    def history(self):
        prev = self
        while prev.prev is not None:
            yield prev.clas
            prev = prev.prev


cdef class Beam:
    def __init__(self, size_t nr_class, size_t width):
        self.nr_class = nr_class
        self.width = width
        self.extensions = [None for i in range(self.width)]
        self.history = []
        self.bests = []

    property extensions:
        def __get__(self):
            return self.extensions

    def fill_from_list(self, list scores):
        self.history = self.extensions
        self.extensions = []
        for i in range(self.width):
            for j in range(self.nr_class):
                self.q.push(Entry(scores[i][j], Candidate(i, j)))

    cdef int fill(self, double** scores):
        self.history = self.extensions
        self.extensions = []
        for i in range(self.width):
            for j in range(self.nr_class):
                self.q.push(Entry(scores[i][j], Candidate(i, j)))

    cpdef Candidate pop(self) except *:
        cdef double score
        cdef Candidate c
        score, c = self.q.top()
        self.extensions.append(Move(score, c.second, self.history[c.first]))
        if len(self.extensions) == 1:
            self.bests.append(self.extensions[-1])
        self.q.pop()
        return c

    def max_violation(self, Beam gold):
        deltas = []
        # It's correct to halt early if gold is longer or shorter --- 
        # we don't need length alignment.
        for i, (best_p, best_g) in enumerate(zip(self.bests, gold.bests)):
            deltas.append((best_p.score + 1 - best_g.score, i))
        delta, i = max(deltas)
        return delta, list(self.bests[i].history), list(gold.bests[i].history)
