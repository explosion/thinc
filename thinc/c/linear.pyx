# cython: infer_types=True
from libc.stdlib cimport calloc, free, realloc
from libc.string cimport memcpy, memset

from flags cimport count_tasks_remaining, usleep, run_task_f
from flags cimport get_input, get_gradient, yield_output, yield_gradient
from params cimport params_s, refresh_params

from openblas cimport *


cdef task_s make_task(flag_t* status, int layer_id, params_s* params,
        float* inputs, float* outputs, float* d_inputs, float* d_outputs,
        int nr_out, int nr_in, int batch_size, int N) nogil:
    args = <args_s*>calloc(1, sizeof(args_s))
    args.status = status
    args.layer_id= layer_id
    args.params = params
    args.X = inputs
    args.Y = outputs
    args.dX = d_inputs
    args.dY = d_outputs
    args.nr_out = nr_out
    args.nr_in = nr_in
    args.max_batch = batch_size
    args.N = N
    args.params = params 
    cdef task_s task
    task.run = <run_task_f>run_task
    task.args = args
    return task


cdef struct args_s:
    flag_t* status
    int layer_id
    params_s* params
    const float* X
    float* Y
    float* dX
    const float* dY
    int nr_out
    int nr_in
    int max_batch
    int N


cdef void* run_task(args_s* args) nogil:
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
    cdef int fwd_todo, bwd_todo
    count_tasks_remaining(&fwd_todo, &bwd_todo, status, N, layer_id)
    if dY == NULL or (dX == NULL and params.gradient == NULL):
        bwd_todo = 0
    cdef int i, fwd_size, bwd_size
    while fwd_todo or bwd_todo:
        fwd_size = 0
        bwd_size = 0
        get_input(&i, &fwd_size, status, layer_id, N, N)
        forward(&Y[i], &X[i],
            params.weights, nr_out, nr_in, fwd_size)
        yield_output(&status[i], fwd_size, layer_id)
        fwd_todo -= fwd_size
        
        if fwd_todo < bwd_todo:
            get_gradient(&i, &bwd_size, status, layer_id, max_batch, N)
            bwd_todo -= bwd_size
            if dX != NULL:
                backprop_inputs(&dX[i], &dY[i],
                    params.weights, nr_out, nr_in, bwd_size)
                yield_gradient(&status[i], bwd_size, layer_id)
            if params.gradient != NULL:
                backprop_params(params.gradient,
                    &dY[i], X, nr_out, nr_in, bwd_size)
                params.nr_grad_upd += bwd_size
                # If we don't have any examples where we've done the fwd pass
                # but haven't backproped, it's safe to pull a new copy of the
                # weights if they're available.
                if bwd_todo == fwd_todo and params.next != NULL:
                    params = refresh_params(params)
        if fwd_size == 0 and bwd_size == 0:
            break
            #usleep(100000) # Sleep for 0.1 seconds if no tasks were ready.


cdef void forward(float* Y,
        const float* X, const float* Wb,
        int nr_in, int nr_out, int N) nogil:
    memset(Y, 0, N*nr_out*sizeof(float))
    simple_gemm(Y, N, nr_out,
        X, N, nr_in, Wb, nr_in, nr_out)
    for i in range(N):
        simple_axpy(&Y[i*nr_out], nr_out, &Wb[nr_in*nr_out], 1.)


cdef void backprop_inputs(float* dX,
        const float* dY, const float* W, int nr_out, int nr_in, int N) nogil:
    memset(dX, 0, N*nr_in*sizeof(float))
    simple_gemm(dX, N, nr_in,
        dY, N, nr_out, W, nr_in, nr_out)


cdef void backprop_params(float* dWb, 
        const float* dY, const float* X, int nr_out, int nr_in, int N) nogil:
    simple_gemm(dWb, nr_in, nr_out,
        X, N, nr_in, dY, N, nr_out)
    for i in range(N):
        simple_axpy(&dWb[nr_in*nr_out],
            nr_out, &dY[i*nr_out], 1.)
