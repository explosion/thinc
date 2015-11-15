cdef enum ReturnCodes:
    ReturnSuccess
    ReturnMismatchX
    ReturnMismatchY


cdef class Vector:
    cdef int add_vec(float* me, const float* you, float scale, int nr_dim) nogil:
        cblas_saxpy(me.nr_y, scale, me, sizeof(me[0]), you, sizeof(you[0]), nr_dim)


cdef class Matrix:
    cdef MatrixC* c

    def __init__(self):
        pass

    @staticmethod
    cdef inline int dot_bias(MatrixC* me, const MatrixC* W, const MatrixC* b) nogil:
        Matrix.sgemv(me, you, 0.0, NULL)
        Matrix.saxpy(me, you, 1.0)

    @staticmethod
    cdef inline int saxpy(MatrixC* me, const MatrixC* you, float scale) nogil:
        cblas_saxpy(me.nr_col, 1.0, me.data, sizeof(me.data[0]), you.data, sizeof(you.data[0]))

    @staticmethod
    cdef inline int sgemv(MatrixC* me, const MatrixC* you, float scale, MatrixC* y) nogil:
        cblas_sgemv(
            'n',
            you.nr_x, you.nr_y, scale, you.data,
            you.nr_x,
            me.data, sizeof(me.data[0]),
            0.0, NULL, 0
        )
 
    @staticmethod
    cdef inline int imul(MatrixC* me, const MatrixC* you, float scale) nogil:
        err = check_data_integrity(me, you)
        if err != ReturnSuccess:
            return err
        
        # Level 1
        if me.nr_x == 1 and you.nr_x == 1:
            cblas_saxpy(me.nr_y, scale_you, me.data, me.stride, you.data, you.stride)
        elif me.nr_x == 1:
            cblas_sgemv(
                'n',
                you.nr_x, you.nr_y, scale, you.data,
                you.nr_x,
                me.data, me.stride,
                0.0, NULL, 0)
        return ReturnSuccess

    @staticmethod
    cdef inline void isub(MatrixC* m1, const MatrixC* m2, float scale) nogil:
        pass

    @staticmethod
    cdef inline void idiv(MatrixC* m1, const MatrixC* m2, float scale) nogil:
        pass

    @staticmethod
    cdef inline void idot(MatrixC* m1, const MatrixC* m2, float scale) nogil:
        pass
