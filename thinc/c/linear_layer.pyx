# cython: infer_types=True
from libc.stdlib cimport calloc, free, realloc
from libc.string cimport memcpy, memset

include "openblas.pyx"
include "flags.pyx"

cdef struct linear_args_s:
    flag_t* status
    int layer_id
    void* params
    const float* X
    float* Y
    float* dX
    const float* dY
    int nr_out
    int nr_in
    int max_batch
    int N


cdef void update_linear(linear_args_s* args) nogil:
    status = args.status
    layer_id = args.layer_id
    X = args.X
    Y = args.Y
    dX = args.dX
    dY = args.dY
    nr_out = args.nr_out
    nr_in = args.nr_in
    max_batch = args.max_batch
    N = args.N
    params = args.params
    cdef const float* W
    cdef const float* b
    cdef float* dW
    cdef float* db
    cdef int i, size
    while tasks_remaining(status, N, layer_id):
        await_input(&i, &size, status, max_batch, N, layer_id)
        get_params(params, &W, &b, &dW, &db, nr_out, nr_in)
        
        forward(&Y[i],
            &X[i], W, b, nr_out, nr_in, size)

        yield_output(&status[i], size, layer_id)
        
        if dY != NULL:
            await_gradient(&status[i], size, layer_id)
            if dX != NULL:
                backprop_inputs(&dX[i], 
                    &dY[i], W, nr_out, nr_in, size)
            if dW != NULL:
                backprop_params(dW, db,
                    &dY[i], X, nr_out, nr_in, size)
            yield_gradient(&status[i], size, layer_id)


cdef void get_params(void* _params, float** W, float** b, float** dW, float** db,
        int nr_in, int nr_out) nogil:
    params = <float**>_params
    cdef float* Wb
    cdef float* dWb
    with gil:
        Wb = params[0]
        dWb = params[1]
    W[0] = Wb
    b[0] = &Wb[nr_in * nr_out]
    dW[0] = dWb
    db[0] = &dWb[nr_in*nr_out]


cdef void forward(float* Y,
        const float* X, const float* W, const float* b,
        int nr_in, int nr_out, int N) nogil:
    memset(Y, 0, N*nr_out*sizeof(float))
    simple_gemm(Y, N, nr_out,
        X, N, nr_in, W, nr_in, nr_out)
    for i in range(N):
        simple_axpy(&Y[i*nr_out], nr_out, b, 1.)


cdef void backprop_inputs(float* dX,
        const float* dY, const float* W, int nr_out, int nr_in, int N) nogil:
    memset(dX, 0, N*nr_in*sizeof(float))
    simple_gemm(dX, N, nr_in,
        dY, N, nr_out, W, nr_in, nr_out)


cdef void backprop_params(float* dW, float* db, 
        const float* dY, const float* X, int nr_out, int nr_in, int N) nogil:
    simple_gemm(dW, nr_in, nr_out,
        X, N, nr_in, dY, N, nr_out)
    for i in range(N):
        simple_axpy(db,
            nr_out, &dY[i*nr_out], 1.)
