cimport blis.cy
from cython.operator cimport dereference as deref
from libcpp.memory cimport make_shared


# Single-precision wrapper for `blis.cy.scalv`
cdef void blis_sscalv(int N, float alpha, float* X, int incX) nogil:
    blis.cy.scalv(blis.cy.NO_CONJUGATE, N, alpha, X, incX)


cdef struct BlasFuncs:
    daxpy_ptr daxpy
    saxpy_ptr saxpy
    sgemm_ptr sgemm
    sscalv_ptr sscalv


cdef class CBlas:
    __slots__ = []

    def __init__(self):
        """Construct a CBlas instance set to use BLIS implementations of the
           supported BLAS functions."""
        cdef BlasFuncs funcs
        funcs.daxpy = blis.cy.daxpy
        funcs.saxpy = blis.cy.saxpy
        funcs.sgemm = blis.cy.sgemm
        funcs.sscalv = blis_sscalv
        self.ptr = make_shared[BlasFuncs](funcs)

cdef daxpy_ptr daxpy(CBlas cblas) nogil:
    return deref(cblas.ptr).daxpy

cdef saxpy_ptr saxpy(CBlas cblas) nogil:
    return deref(cblas.ptr).saxpy

cdef sgemm_ptr sgemm(CBlas cblas) nogil:
    return deref(cblas.ptr).sgemm

cdef sscalv_ptr sscalv(CBlas cblas) nogil:
    return deref(cblas.ptr).sscalv

cdef void set_daxpy(CBlas cblas, daxpy_ptr daxpy) nogil:
    deref(cblas.ptr).daxpy = daxpy

cdef void set_saxpy(CBlas cblas, saxpy_ptr saxpy) nogil:
    deref(cblas.ptr).saxpy = saxpy

cdef void set_sgemm(CBlas cblas, sgemm_ptr sgemm) nogil:
    deref(cblas.ptr).sgemm = sgemm

cdef void set_sscalv(CBlas cblas, sscalv_ptr sscalv) nogil:
    deref(cblas.ptr).sscalv = sscalv
