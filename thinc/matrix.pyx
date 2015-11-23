from __future__ import print_function


cdef class Matrix:
    def __init__(self, int nr_row, int nr_col):
        self.mem = Pool()
        self.c = Matrix.initC(self.mem, nr_row, nr_col)

    def get(self, int row, int col):
        return Matrix.getC(self.c, row, col)

    def set(self, int row, int col, float value):
        Matrix.setC(self.c, row, col, value)

    def iadd(self, Matrix you, float scale=1.0):
        Matrix.iaddC(self.c, you.c, scale)
    
    def dot_bias(self, Matrix x, Matrix W, Matrix b):
        Matrix.dot_biasC(self.c, x.c, W.c, b.c)
