# cython: profile=False
cimport blis.cy
from cython.operator cimport dereference as deref
from libcpp.memory cimport make_shared


# Single- and double-precision wrappers for `blis.cy.scalv`
cdef void blis_sscal(int N, float alpha, float* X, int incX) nogil:
    blis.cy.scalv(blis.cy.NO_CONJUGATE, N, alpha, X, incX)

cdef void blis_dscal(int N, double alpha, double* X, int incX) nogil:
    blis.cy.scalv(blis.cy.NO_CONJUGATE, N, alpha, X, incX)


cdef struct BlasFuncs:
    daxpy_ptr daxpy
    saxpy_ptr saxpy
    sgemm_ptr sgemm
    dgemm_ptr dgemm
    sscal_ptr sscal
    dscal_ptr dscal


cdef class CBlas:
    __slots__ = []

    def __init__(self):
        """Construct a CBlas instance set to use BLIS implementations of the
           supported BLAS functions."""
        cdef BlasFuncs funcs
        funcs.daxpy = blis.cy.daxpy
        funcs.saxpy = blis.cy.saxpy
        funcs.sgemm = blis.cy.sgemm
        funcs.dgemm = blis.cy.dgemm
        funcs.sscal = blis_sscal
        funcs.dscal = blis_dscal
        self.ptr = make_shared[BlasFuncs](funcs)

cdef daxpy_ptr daxpy(CBlas cblas) nogil:
    return deref(cblas.ptr).daxpy

cdef saxpy_ptr saxpy(CBlas cblas) nogil:
    return deref(cblas.ptr).saxpy

cdef sgemm_ptr sgemm(CBlas cblas) nogil:
    return deref(cblas.ptr).sgemm

cdef dgemm_ptr dgemm(CBlas cblas) nogil:
    return deref(cblas.ptr).dgemm

cdef sscal_ptr sscal(CBlas cblas) nogil:
    return deref(cblas.ptr).sscal

cdef dscal_ptr dscal(CBlas cblas) nogil:
    return deref(cblas.ptr).dscal

cdef void set_daxpy(CBlas cblas, daxpy_ptr daxpy) nogil:
    deref(cblas.ptr).daxpy = daxpy

cdef void set_saxpy(CBlas cblas, saxpy_ptr saxpy) nogil:
    deref(cblas.ptr).saxpy = saxpy

cdef void set_sgemm(CBlas cblas, sgemm_ptr sgemm) nogil:
    deref(cblas.ptr).sgemm = sgemm

cdef void set_dgemm(CBlas cblas, dgemm_ptr dgemm) nogil:
    deref(cblas.ptr).dgemm = dgemm

cdef void set_sscal(CBlas cblas, sscal_ptr sscal) nogil:
    deref(cblas.ptr).sscal = sscal

cdef void set_dscal(CBlas cblas, dscal_ptr dscal) nogil:
    deref(cblas.ptr).dscal = dscal
