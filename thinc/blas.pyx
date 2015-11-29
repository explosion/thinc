cimport cython


cdef class Matrix:
    @classmethod
    def from_array(cls, ndarray):
        assert len(ndarray.shape) <= 2
        if len(ndarray.shape) == 1:
            nr_row = 1
            nr_col = ndarray.shape[0]
        else:
            nr_row = ndarray.shape[0]
            nr_col = ndarray.shape[1]
        cdef Matrix self = cls(nr_row, nr_col)
        cdef int i
        cdef weight_t value
        for i, value in enumerate(ndarray.flatten()):
            self.data[i] = value
        return self

    def __init__(self, nr_row, nr_col):
        self.mem = Pool()
        self.nr_row = nr_row
        self.nr_col = nr_col
        self.data = <weight_t*>self.mem.alloc(nr_row * nr_col, sizeof(weight_t))

    def __iter__(self):
        cdef int i
        for i in range(self.nr_row * self.nr_col):
            yield self.data[i]
    
    def __getitem__(self, int i):
        return self.data[i]

    def __richcmp__(Matrix self, other, int cmp_type):
        # < 0
        # == 2
        # > 4
        # <= 1
        # != 3
        # >= 5
        cdef int i
        if cmp_type == 0:
            raise NotImplementedError
        elif cmp_type == 2 or cmp_type == 3:
            for i in range(self.nr_row * self.nr_col):
                if other[i] != self.data[i]:
                    return False if cmp_type == 2 else True
            return True if cmp_type == 2 else False
        elif cmp_type == 4:
            raise NotImplementedError
        elif cmp_type == 1:        
            raise NotImplementedError
        elif cmp_type == 5:
            raise NotImplementedError
        else:
            raise ValueError

    def __iadd__(self, other):
        cdef Matrix other_mat
        if isinstance(other, Matrix):
            other_mat = other
            assert self.nr_row == other_mat.nr_row
            assert self.nr_col == other_mat.nr_col
            if self.nr_row == 1:
                VecVec.add_i(self.data, other_mat.data, 1.0, self.nr_col)
            else:
                MatMat.add_i(self.data, other_mat.data, self.nr_row, self.nr_col)
        elif isinstance(other, int) or isinstance(other, float):
            Vec.add_i(self.data, other, self.nr_row * self.nr_col)
        else:
            raise ValueError
        return self

    def __imul__(self, other):
        cdef Matrix other_mat
        if isinstance(other, Matrix):
            other_mat = other
            assert self.nr_row == other.nr_row
            assert self.nr_col == other.nr_col
            if self.nr_row == 1:
                VecVec.mul_i(self.data, other_mat.data, self.nr_row)
            else:
                MatMat.mul_i(self.data, other_mat.data, self.nr_row, self.nr_col)
        elif isinstance(other, int) or isinstance(other, float):
            Vec.mul_i(self.data, other, self.nr_row * self.nr_col)
        else:
            raise ValueError
        return self

    def __idiv__(self, weight_t scalar):
        Vec.div_i(self.data, scalar, self.nr_row * self.nr_col)
        return self
    
    def __ipow__(self, weight_t scalar):
        Vec.pow_i(self.data, scalar, self.nr_row * self.nr_col)
        return self

    def max(self):
        return Vec.max(self.data, self.nr_row * self.nr_col)

    def sum(self):
        return Vec.sum(self.data, self.nr_row * self.nr_col)

    def exp(self):
        Vec.exp_i(self.data, self.nr_row * self.nr_col)
        return self

    def dot(self, Matrix x):
        assert self.nr_col == x.nr_col
        cdef Matrix output = Matrix(1, self.nr_row)
        MatVec.dot(output.data, self.data, x.data, self.nr_row, self.nr_col)
        return output

    def dot_bias(self, Matrix W, Matrix b):
        assert self.nr_row == 1
        assert b.nr_row == 1
        assert W.nr_col == self.nr_col == b.nr_col
        cdef Matrix output = Matrix(1, self.nr_row)

        MatVec.dot(output.data, W.data, self.data, W.nr_row, W.nr_col)
        VecVec.add_i(output.data, b.data, 1.0, self.nr_col)
        return output

    def add_outer(self, Matrix x, Matrix y):
        assert x.nr_row == 1
        assert y.nr_row == 1
        assert self.nr_row == x.nr_col
        assert self.nr_col == y.nr_col
        MatMat.add_outer_i(self.data, x.data, y.data, self.nr_row, self.nr_col)

    def set(self, int32_t row, int32_t col, weight_t value):
        if row >= self.nr_row or col >= self.nr_col:
            raise IndexError

        self.data[row * self.nr_row + col] = value

    def get(self, int32_t row, int32_t col):
        if row >= self.nr_row or col >= self.nr_col:
            raise IndexError

        return self.data[row * self.nr_row + col]



