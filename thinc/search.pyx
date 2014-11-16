cimport cython
from libc.string cimport memset, memcpy

from cymem.cymem cimport Pool


cdef class Beam:
    def __init__(self, class_t nr_class, class_t width):
        assert nr_class != 0
        assert width != 0
        self.nr_class = nr_class
        self.width = width
        self.size = 1
        self.mem = Pool()
        self._parents = <_State*>self.mem.alloc(self.width, sizeof(_State))
        self._states = <_State*>self.mem.alloc(self.width, sizeof(_State))

        self.scores = <weight_t**>self.mem.alloc(self.width, sizeof(weight_t*))
        self.is_valid = <bint**>self.mem.alloc(self.width, sizeof(bint*))
        self.costs = <int**>self.mem.alloc(self.width, sizeof(int*))
        cdef int i
        for i in range(self.width):
            self.scores[i] = <weight_t*>self.mem.alloc(self.nr_class, sizeof(weight_t))
            self.is_valid[i] = <bint*>self.mem.alloc(self.nr_class, sizeof(bint))
            self.costs[i] = <int*>self.mem.alloc(self.nr_class, sizeof(int))

    property score:
        def __get__(self):
            return self.q.top().first

    cpdef int set_cell(self, int i, int j, weight_t score, bint is_valid, int cost) except -1:
        self.scores[i][j] = score
        self.is_valid[i][j] = is_valid
        self.costs[i][j] = cost
 
    cdef int set_row(self, int i, weight_t* scores, bint* is_valid, int* costs) except -1:
        cdef int j
        for j in range(self.nr_class):
            self.scores[i][j] = scores[j]
            self.is_valid[i][j] = is_valid[j]
            self.costs[i][j] = costs[j]

    cdef int set_table(self, weight_t** scores, bint** is_valid, int** costs) except -1:
        cdef int i, j
        for i in range(self.width):
            memcpy(self.scores[i], scores[i], sizeof(weight_t) * self.nr_class)
            memcpy(self.is_valid[i], is_valid[i], sizeof(bint) * self.nr_class)
            memcpy(self.costs[i], costs[i], sizeof(int) * self.nr_class)
    
    cdef void* at(self, int i):
        return self._states[i].content

    cdef int initialize(self, init_func_t init_func, int n, void* extra_args) except -1:
        for i in range(self.width):
            self._states[i].content = init_func(self.mem, n, extra_args)
            self._parents[i].content = init_func(self.mem, n, extra_args)

    cdef int advance(self, trans_func_t transition_func, void* extra_args) except -1:
        cdef weight_t** scores = self.scores
        cdef bint** is_valid = self.is_valid
        cdef int** costs = self.costs

        self._fill(scores, is_valid)
        # For a beam of width k, we only ever need 2k state objects. How?
        # Each transition takes a parent and a class and produces a new state.
        # So, we don't need the whole history --- just the parent. So at
        # each step, we take a parent, and apply one or more extensions to
        # it.
        self._parents, self._states = self._states, self._parents
        cdef weight_t score
        cdef int p_i
        cdef int i = 0
        cdef class_t clas
        cdef _State* parent
        cdef _State* state
        while i < self.width and not self.q.empty():
            score, (p_i, clas) = self.q.top()
            self.q.pop()
            # Indicates terminal state reached; i.e. state is done
            if clas == 0:
                # Now parent will not be changed, so we don't have to copy.
                assert self._parents[p_i].is_done
                self._states[i] = self._parents[p_i]
                self._states[i].score = score
                continue
            parent = &self._parents[p_i]
            state = &self._states[i]
            # The supplied transition function should adjust the destination
            # state to be the result of applying the class to the source state
            transition_func(state.content, parent.content, clas, extra_args)
            state.score = parent.score + scores[p_i][clas]
            state.loss = parent.loss + costs[p_i][clas]
            #state.hist[state.t] = clas
            #state.t += 1
            i += 1
        self.size = i
        for i in range(self.width):
            memset(self.scores[i], 0, sizeof(weight_t) * self.nr_class)
            memset(self.is_valid[i], False, sizeof(bint) * self.nr_class)
            memset(self.costs[i], 0, sizeof(int) * self.nr_class)

    cdef int check_done(self, finish_func_t finish_func, void* extra_args) except -1:
        cdef int i
        self.is_done = True
        for i in range(self.size):
            self._states[i].is_done = finish_func(&self._states[i].content, extra_args)
            if not self._states[i].is_done:
                self.is_done = False

    cdef int _fill(self, weight_t** scores, bint** is_valid) except -1:
        """Populate the queue from a k * n matrix of scores, where k is the
        beam-width, and n is the number of classes.
        """
        cdef Candidate candidate
        cdef Entry entry
        cdef weight_t score
        while not self.q.empty():
            self.q.pop()
        cdef _State* s
        for i in range(self.width):
            s = &self._states[i]
            if s.is_done:
                # Update score by path average, following TACL '13 paper.
                entry = Entry(s.score + (s.score / s.t), Candidate(i, 0))
                self.q.push(entry)
                continue
            for j in range(self.nr_class):
                if is_valid[i][j]:
                    entry = Entry(scores[i][j], Candidate(i, j))
                    self.q.push(entry)



cdef class MaxViolation:
    def __init__(self):
        self.delta = -1
        self.n = 0
        self.cost = 0
        self.p_hist = []
        self.g_hist = []

    cpdef int check(self, Beam pred, Beam gold) except -1:
        cdef _State* p = &pred._states[0]
        cdef _State* g = &gold._states[0]
        cdef weight_t d = (p.score + 1) - g.score
        if p.loss and d > self.delta:
            self.loss = p.loss
            self.delta = d
            self.p_hist = [p.hist[i] for i in range(p.t)]
            self.g_hist = [g.hist[i] for i in range(g.t)]
