from __future__ cimport division
cimport cython
from libc.string cimport memset, memcpy

from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap
from .typedefs cimport hash_t


cdef class Beam:
    def __init__(self, class_t nr_class, class_t width):
        assert nr_class != 0
        assert width != 0
        self.nr_class = nr_class
        self.width = width
        self.size = 1
        self.t = 0
        self.mem = Pool()
        self._parents = <_State*>self.mem.alloc(self.width, sizeof(_State))
        self._states = <_State*>self.mem.alloc(self.width, sizeof(_State))
        cdef int i
        self.histories = [[] for i in range(self.width)]
        self._parent_histories = [[] for i in range(self.width)]

        self.scores = <weight_t**>self.mem.alloc(self.width, sizeof(weight_t*))
        self.is_valid = <bint**>self.mem.alloc(self.width, sizeof(bint*))
        self.costs = <int**>self.mem.alloc(self.width, sizeof(int*))
        for i in range(self.width):
            self.scores[i] = <weight_t*>self.mem.alloc(self.nr_class, sizeof(weight_t))
            self.is_valid[i] = <bint*>self.mem.alloc(self.nr_class, sizeof(bint))
            self.costs[i] = <int*>self.mem.alloc(self.nr_class, sizeof(int))

    property score:
        def __get__(self):
            return self._states[0].score

    property loss:
        def __get__(self):
            return self._states[0].loss
 
    cdef int set_row(self, int i, const weight_t* scores, const bint* is_valid,
                     const int* costs) except -1:
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
    
    cdef int initialize(self, init_func_t init_func, int n, void* extra_args) except -1:
        for i in range(self.width):
            self._states[i].content = init_func(self.mem, n, extra_args)
            self._parents[i].content = init_func(self.mem, n, extra_args)

    @cython.cdivision(True)
    cdef int advance(self, trans_func_t transition_func, hash_func_t hash_func,
                     void* extra_args) except -1:
        cdef weight_t** scores = self.scores
        cdef bint** is_valid = self.is_valid
        cdef int** costs = self.costs

        cdef Queue* q = new Queue()
        self._fill(q, scores, is_valid)
        # For a beam of width k, we only ever need 2k state objects. How?
        # Each transition takes a parent and a class and produces a new state.
        # So, we don't need the whole history --- just the parent. So at
        # each step, we take a parent, and apply one or more extensions to
        # it.
        self._parents, self._states = self._states, self._parents
        self._parent_histories, self.histories = self.histories, self._parent_histories
        cdef weight_t score
        cdef int p_i
        cdef int i = 0
        cdef class_t clas
        cdef _State* parent
        cdef _State* state
        cdef hash_t key
        cdef PreshMap seen_states = PreshMap(self.width)
        cdef size_t is_seen
        while i < self.width and not q.empty():
            data = q.top()
            p_i = data.second / self.nr_class
            clas = data.second % self.nr_class
            score = data.first
            q.pop()
            parent = &self._parents[p_i]
            # Indicates terminal state reached; i.e. state is done
            if parent.is_done:
                # Now parent will not be changed, so we don't have to copy.
                self._states[i] = parent[0]
                self._states[i].score = score
                i += 1
            else:
                state = &self._states[i]
                # The supplied transition function should adjust the destination
                # state to be the result of applying the class to the source state
                transition_func(state.content, parent.content, clas, extra_args)
                key = hash_func(state.content, extra_args) if hash_func is not NULL else 0
                is_seen = <size_t>seen_states.get(key)
                if key == 0 or not is_seen:
                    if key != 0:
                        seen_states.set(key, <void*>1)
                    state.score = score
                    state.loss = parent.loss + costs[p_i][clas]
                    self.histories[i] = list(self._parent_histories[p_i])
                    self.histories[i].append(clas)
                    i += 1
        del q
        self.size = i
        assert self.size >= 1
        for i in range(self.width):
            memset(self.scores[i], 0, sizeof(weight_t) * self.nr_class)
            memset(self.is_valid[i], False, sizeof(bint) * self.nr_class)
            memset(self.costs[i], 0, sizeof(int) * self.nr_class)
        self.t += 1

    cdef int check_done(self, finish_func_t finish_func, void* extra_args) except -1:
        cdef int i
        for i in range(self.size):
            if not self._states[i].is_done:
                self._states[i].is_done = finish_func(self._states[i].content, extra_args)
                if not self._states[i].is_done:
                    self.is_done = False
                    break
        else:
            self.is_done = True

    @cython.cdivision(True)
    cdef int _fill(self, Queue* q, weight_t** scores, bint** is_valid) except -1:
        """Populate the queue from a k * n matrix of scores, where k is the
        beam-width, and n is the number of classes.
        """
        cdef Entry entry
        cdef weight_t score
        cdef _State* s
        cdef int i, j, move_id
        assert self.size >= 1
        for i in range(self.size):
            s = &self._states[i]
            move_id = i * self.nr_class
            if s.is_done:
                # Update score by path average, following TACL '13 paper.
                if self.histories[i]:
                    entry.first = s.score + (s.score / self.t)
                else:
                    entry.first = s.score
                entry.second = move_id
                q.push(entry)
            else:
                for j in range(self.nr_class):
                    if is_valid[i][j]:
                        entry.first = s.score + scores[i][j]
                        entry.second = move_id + j
                        q.push(entry)


cdef class MaxViolation:
    def __init__(self):
        self.delta = -1
        self.cost = 0
        self.p_hist = []
        self.g_hist = []

    cpdef int check(self, Beam pred, Beam gold) except -1:
        cdef _State* p = &pred._states[0]
        cdef _State* g = &gold._states[0]
        cdef weight_t d = p.score - g.score
        if p.loss >= 1 and (self.cost == 0 or d > self.delta):
            self.cost = p.loss
            self.delta = d
            self.p_hist = list(pred.histories[0])
            self.g_hist = list(gold.histories[0])
