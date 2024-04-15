cimport numpy as np
from libc.stdint cimport uintptr_t

import numpy


cpdef np.ndarray gemm(float[:, ::1] A, float[:, ::1] B,
                      bint trans1=False, bint trans2=False,
                      np.ndarray out=None):
    cdef int nM = A.shape[0] if not trans1 else A.shape[1]
    cdef int nK = A.shape[1] if not trans1 else A.shape[0]
    cdef int nK_b = B.shape[0] if not trans2 else B.shape[1]
    cdef int nN = B.shape[1] if not trans2 else B.shape[0]

    cdef float[:, ::1] C = out

    if out is None:
        out = numpy.empty((nM, nN), dtype="f")
        C = out
    else:
        if C.shape[0] != nM or C.shape[1] != nN:
            msg = "Shape mismatch for output matrix, was: (%d, %d), expected (%d, %d)"
            raise ValueError(msg % (C.shape[0], C.shape[1], nM, nN))


    if nK != nK_b:
        msg = "Shape mismatch for gemm: (%d, %d), (%d, %d)"
        raise ValueError(msg % (nM, nK, nK_b, nN))

    if nM == 0 or nK == 0 or nN == 0:
        return out

    cblas_sgemm(
        CblasRowMajor,
        CblasTrans if trans1 else CblasNoTrans,
        CblasTrans if trans2 else CblasNoTrans,
        nM,
        nN,
        nK,
        1.0,
        &A[0, 0],
        A.shape[1],
        &B[0, 0],
        B.shape[1],
        0.0,
        &C[0, 0],
        C.shape[1]
    )
    return out


cdef void sgemm(bint TransA, bint TransB, int M, int N, int K,
                    float alpha, const float* A, int lda, const float *B,
                    int ldb, float beta, float* C, int ldc) nogil:
    cblas_sgemm(
        CblasRowMajor,
        CblasTrans if TransA else CblasNoTrans,
        CblasTrans if TransB else CblasNoTrans,
        M,
        N,
        K,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc
    )


cdef void saxpy(int N, float alpha, const float* X, int incX,
                float *Y, int incY) nogil:
    cblas_saxpy(N, alpha, X, incX, Y, incY)
