from libc.stdlib cimport calloc, free


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
        cdef double** c_scores = <double**>calloc(len(scores), sizeof(double*))
        for i, clas_scores in enumerate(scores):
            c_scores[i] = <double*>calloc(len(clas_scores), sizeof(double))
            for j, score in enumerate(clas_scores):
                c_scores[i][j] = score
        self.fill(c_scores)
        for i in range(len(scores)):
            free(c_scores[i])
        free(c_scores)

    cdef int fill(self, double** scores):
        cdef Candidate candidate
        cdef Entry entry
        cdef double score
        cdef size_t addr
        self.history = self.extensions
        self.extensions = []
        while not self.q.empty():
            self.q.pop()
        for i in range(self.width):
            for j in range(self.nr_class):
                entry = Entry(scores[i][j], Candidate(i, j))
                self.q.push(entry)

    cpdef pair[size_t, size_t] pop(self) except *:
        if self.q.empty():
            raise StopIteration
        cdef double score
        cdef size_t addr
        score, (parent, clas) = self.q.top()
        self.extensions.append(Move(score, clas, self.history[parent]))
        cdef size_t parent = c.parent
        cdef size_t clas = c.clas
        if len(self.extensions) == 1:
            self.bests.append(self.extensions[-1])
        self.q.pop()
        return pair[size_t, size_t](parent, clas)

    def max_violation(self, Beam gold):
        cdef Move best_p
        cdef Move best_g
        deltas = []
        # It's correct to halt early if gold is longer or shorter --- 
        # we don't need length alignment.
        for i, (best_p, best_g) in enumerate(zip(self.bests, gold.bests)):
            if best_p.cost >= 1:
                deltas.append((best_p.score + 1 - best_g.score, i))
        if not deltas:
            return 0, [], []
        delta, i = max(deltas)
        best_p = self.bests[i]
        best_g = gold.bests[i]
        pred_moves = list(best_p.history())
        pred_moves.reverse()
        gold_moves = list(best_g.history())
        gold_moves.reverse()
        return delta, pred_moves, gold_moves
