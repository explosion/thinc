# cython: infer_types=True
"""
Each layer is a function (self, inputs) --> StateC
The StateC struct defines a callback backward (void*, d_outputs) --> d_inputs

Functions take ownership of their inputs, caller takes ownership of the output
So: layers can reuse their input for output, otherwise they must free it.
"""
cimport cython.parallel
cimport numpy as np
from libc.stdlib cimport calloc, realloc, free
from libc.string cimport memcpy, memset
include "tensor.pyx"
import numpy
import numpy.random
from timeit import default_timer as timer
from libcpp.vector cimport vector

from posix.signal cimport sigset_t

cdef extern from "<pthread.h>" nogil:
    # POSIX says this might be a struct, but CPython (and llfuse)
    # rely on it being an integer.
    ctypedef int pthread_t

    ctypedef struct pthread_attr_t:
        pass
    ctypedef struct pthread_mutexattr_t:
        pass
    ctypedef struct pthread_mutex_t:
       pass

    enum:
        PTHREAD_CANCEL_ENABLE
        PTHREAD_CANCEL_DISABLE

    int pthread_cancel(pthread_t thread)
    int pthread_setcancelstate(int state, int *oldstate)
    pthread_t pthread_self()
    int pthread_sigmask(int how, sigset_t *set, sigset_t *oldset)
    int pthread_equal(pthread_t t1, pthread_t t2)
    int pthread_create(pthread_t *thread, pthread_attr_t *attr,
                       void *(*start_routine) (void *), void *arg)
    int pthread_join(pthread_t thread, void **retval)
    int pthread_kill(pthread_t thread, int sig)

    int pthread_mutex_init(pthread_mutex_t *mutex, pthread_mutexattr_t *mutexattr)
    int pthread_mutex_lock(pthread_mutex_t *mutex)
    int pthread_mutex_unlock(pthread_mutex_t *mutex)


cdef struct callback_s:
    void* call(callback_s* self, void* args) nogil
    void* _

ctypedef void* (*callback_f)(callback_s* self, void* args) nogil

cdef struct result_s:
    void* outputs
    callback_s backprop


ctypedef result_s (*forward_f)(const layer_s* self, void* inputs) nogil


cdef struct layer_s:
    void free(layer_s* self) nogil
    forward_f forward
    void* _

########
# Linear
########


cdef struct linear_result_s:
    Tensor2d* outputs
    callback_s backprop


cdef struct linear_backprop_state_s:
    Tensor2d W
    Tensor2d X
    Tensor2d dWb


cdef struct linear_params_s:
    Tensor2d Wb
    Tensor2d dWb


cdef struct linear_layer_s:
    void free(linear_layer_s self) nogil
    linear_result_s forward(const linear_layer_s* self, Tensor2d* inputs) nogil
    linear_params_s* _


cdef linear_layer_s new_linear_layer() nogil:
    cdef linear_layer_s self
    self._ = <linear_params_s*>calloc(1, sizeof(linear_params_s))
    self.forward = call_linear_forward
    self.free = free_linear
    return self


cdef void free_linear(linear_layer_s self) nogil:
    if self._ == NULL:
        return
    free_tensor(self._.Wb)
    free_tensor(self._.dWb)
    free(self._)
    self._ = NULL


cdef linear_result_s call_linear_forward(const linear_layer_s* self, Tensor2d* X) nogil:
    cdef Tensor2d Y
    memset(&Y, 0, sizeof(Y))
    ensure_alloc(&Y, X.n0, self._.Wb.n1)
    
    linear(Y, X[0], self._.Wb)

    state = <linear_backprop_state_s*>calloc(1, sizeof(linear_backprop_state_s))
    state.W = self._.Wb
    state.W.n0 -= 1
    state.X = X[0]
    state.dWb = self._.dWb
    # Take over the struct for the output, to avoid allocating another
    X[0] = Y
    cdef linear_result_s r
    r.outputs = X
    r.backprop.call = <callback_f>call_linear_backward
    r.backprop._ = state
    return r


cdef Tensor2d* call_linear_backward(callback_s* self, Tensor2d* dY) nogil:
    s = <linear_backprop_state_s*>self._
    cdef Tensor2d dX, dW
    cdef Tensor1d db
    ensure_alloc(&dX, s.X.n0, s.W.n1)
    divide_Wb(&dW, &db, s.dWb)
    gemm(dX, dY[0], s.W)
    gemm(dW, s.X, dY[0])
    for i in range(dY.n0):
        simple_axpy(db.data, dY.n1, &dY.data[i*dY.n1], 1.)
    dY[0] = dX
    free(self._)
    self._ = NULL
    return dY
 

#########
# Chain
#########

#cdef layer_s chain(layer_s first, layer_s second) nogil:
#    cdef layer_s output
#    output.forward = call_chain_forward
#    output.params_size = chain_params_size
#    output.set_params = chain_set_params
#    children = <layer_s*>calloc(2, sizeof(layer_s))
#    output._ = <void*>children
#    return output
#
#
#cdef result_s call_chain_forward(layer_s* self, void* x0) nogil:
#    layers = <layer_s*>self._
#
#    r1 = layers[0].forward(&layers[0], x0)
#    r2 = layers[1].forward(&layers[1], r1.outputs)
#    get_d_inputs = <callback_s*>calloc(2, sizeof(callback_s))
#    callbacks[0] = r1.backprop
#    callbacks[1] = r2.backprop
#
#    cdef result_s r
#    r.outputs = r2.outputs
#    r.backprop.call = call_chain_backward
#    r.backprop._ = callbacks
#    return r
#
#
#cdef void* call_chain_backward(callback_s* self, void* d_x2,
#        callback_s* send_d_params) nogil:
#    children = <callback_s*>self._
#    d_x1 = children[1].call(&children[1], d_x2, &send_d_params[0])
#    d_x0 = children[0].call(&children[0], d_x1, &send_d_params[1])
#    self.free(self)
#    return d_x0
#
cdef void resize_linear(linear_layer_s* layer, int nr_out, int nr_in) nogil:
    Wb = <Tensor2d*>layer._
    ensure_alloc(Wb, nr_in+1, nr_out)


#cdef layer_s xavier_init(layer_s* layer) with gil:
#    Wb = <Tensor2d*>layer._
#
#

cdef object struct2py(Tensor2d tensor):
    cdef np.ndarray arr = numpy.zeros((tensor.n0, tensor.n1), dtype='f')
    memcpy(arr.data, tensor.data, tensor.n0*tensor.n1*sizeof(float))
    free_tensor(tensor)
    return arr


cdef Tensor2d py2struct(np.ndarray arr):
    cdef Tensor2d tensor
    memset(&tensor, 0, sizeof(tensor))
    ensure_alloc(&tensor, arr.shape[0], arr.shape[1])
    memcpy(tensor.data, arr.data, tensor.n0*tensor.n1*sizeof(float))
    return tensor


cdef class Linear:
    cdef linear_layer_s c

    def __init__(self, int nO=0, int nI=0):
        self.c = new_linear_layer()
        resize_linear(&self.c, nO, nI)

    def __dealloc__(self):
        self.c.free(self.c)

    def __call__(self, np.ndarray X):
        cdef Tensor2d tensor = py2struct(X)
        r = self.c.forward(&self.c, &tensor)
        Y = struct2py(r.outputs[0])
        cdef _BackpropLinear backprop = _BackpropLinear()
        backprop.c = r.backprop
        return Y, backprop

    def set_params(self, *, np.ndarray Wb=None, np.ndarray W=None, np.ndarray b=None):
        memcpy(self.c._.Wb.data, <float*>W.data, W.size * sizeof(float))
        memcpy(&self.c._.Wb.data[W.size], <float*>b.data, b.size * sizeof(float))


cdef class _BackpropLinear:
    cdef callback_s c
    
    def __call__(self, np.ndarray dY):
        cdef Tensor2d tensor_dY = py2struct(dY)
        tensor_dX = <Tensor2d*>self.c.call(&self.c, <void*>&tensor_dY)
        arr = struct2py(tensor_dX[0])
        free(tensor_dX)
        return arr


def get_data(batch_size, nr_in, nr_out, nr_batch):
    Xs = [numpy.zeros((batch_size, nr_in), dtype='f') for _ in range(nr_batch)]
    W = numpy.zeros((nr_in, nr_out), dtype='f')
    b = numpy.zeros((nr_out,), dtype='f')
    for X in Xs:
        X += numpy.random.uniform(-0.2, 0.2, X.shape)
    W += numpy.random.uniform(-0.2, 0.2, W.shape)
    b += numpy.random.uniform(-0.2, 0.2, b.shape)
    return Xs, W, b


def time_numpy(Xs, W, b, nr_rep):
    start = timer()
    for i in range(nr_rep):
        numpy.random.shuffle(Xs)
        for X in Xs:
            Y = numpy.dot(X, W) + b
    end = timer()
    return end-start


def time_cy1(Xs, model, nr_rep):
    start = timer()
    for i in range(nr_rep):
        numpy.random.shuffle(Xs)
        for X in Xs:
            Y = model(X)
    end = timer()
    return end-start


def time_cy2(Xs, Linear model, nr_rep):
    cdef np.ndarray X
    cdef vector[Tensor2d] X_ptrs
    cdef Tensor2d X_ptr
    for X in Xs:
        X_ptr = py2struct(X)
        X_ptrs.push_back(X_ptr)
    start = timer()
    cdef int i, j
    for i in range(nr_rep):
        with nogil:
            for j in range(X_ptrs.size()):
                r = <linear_result_s>model.c.forward(&model.c, &X_ptrs[i])
                free_tensor(r.outputs[0])
    end = timer()
    return end-start


def main(batch_size=32, nr_out=32, nr_in=32, nr_rep=1):
    nr_batch = 1000000//batch_size
    Xs, W, b = get_data(batch_size=batch_size, nr_out=nr_out, nr_in=nr_in, nr_batch=nr_batch)
    np_time = time_numpy(Xs, W, b, nr_rep=10)
    print(np_time)
    model = Linear(nr_out, nr_in)
    model.set_params(W=W, b=b)
    cy_time = time_cy1(Xs, model, nr_rep=10)
    print(cy_time)
    cy_time = time_cy2(Xs, model, nr_rep=10)
    print(cy_time)
    #Y, backprop = model(numpy.ones((2, 3), dtype='f'))
    #print(Y)

    #for X, y in data:
    #    yh, backprop = model(X)
    #    dX, update = backprop(yh-y)
    #    model = update(model)

if __name__ == '__main__':
    main()
