cdef class Matrix:
    def __init__(self, nr_row, nr_col):
        self.mem = Pool()
        self.nr_row = nr_row
        self.nr_col = nr_col
        self.data = <weight_t*>self.mem.alloc(nr_row * nr_col, sizeof(weight_t))

    def __add__(self, Matrix other):
        pass

    def __iadd__(self, Matrix other):
        assert self.nr_row == other.nr_row
        assert self.nr_col == other.nr_col
        MatMat.add_i(self.data, other.data, self.nr_row, self.nr_col)
        return self

    def __imul__(self, Matrix other):
        assert self.nr_row == other.nr_row
        assert self.nr_col == other.nr_col
        MatMat.mul_i(self.data, other.data, self.nr_row, self.nr_col)
        return self

    def dot_bias(self, Matrix W, Matrix b):
        assert self.nr_row == 1
        assert b.nr_row == 1
        assert W.nr_col == self.nr_col == b.nr_col
        cdef Matrix output = Matrix(self.nr_row, self.nr_col)

        MatVec.dot(output.data, W.data, self.data, W.nr_row, W.nr_col)
        VecVec.add_i(output.data, b.data, 1.0, self.nr_col)
        return output

    def set(self, int32_t row, int32_t col, weight_t value):
        if row >= self.nr_row or col >= self.nr_col:
            raise IndexError

        self.data[row * self.nr_row + col] = value

    def get(self, int32_t row, int32_t col):
        if row >= self.nr_row or col >= self.nr_col:
            raise IndexError

        return self.data[row * self.nr_row + col]



