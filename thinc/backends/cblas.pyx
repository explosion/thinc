cimport blis.cy
from cython.operator cimport dereference as deref
from libcpp.memory cimport make_shared


cdef struct BlasFuncs:
    daxpy_ptr daxpy
    saxpy_ptr saxpy
    sgemm_ptr sgemm
    dgemm_ptr dgemm


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
        self.ptr = make_shared[BlasFuncs](funcs)

cdef daxpy_ptr daxpy(CBlas cblas) nogil:
    return deref(cblas.ptr).daxpy

cdef saxpy_ptr saxpy(CBlas cblas) nogil:
    return deref(cblas.ptr).saxpy

cdef sgemm_ptr sgemm(CBlas cblas) nogil:
    return deref(cblas.ptr).sgemm

cdef dgemm_ptr dgemm(CBlas cblas) nogil:
    return deref(cblas.ptr).dgemm

cdef void set_daxpy(CBlas cblas, daxpy_ptr daxpy) nogil:
    deref(cblas.ptr).daxpy = daxpy

cdef void set_saxpy(CBlas cblas, saxpy_ptr saxpy) nogil:
    deref(cblas.ptr).saxpy = saxpy

cdef void set_sgemm(CBlas cblas, sgemm_ptr sgemm) nogil:
    deref(cblas.ptr).sgemm = sgemm

cdef void set_dgemm(CBlas cblas, dgemm_ptr dgemm) nogil:
    deref(cblas.ptr).dgemm = dgemm