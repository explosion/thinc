# cython: infer_types=True
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from flags cimport flag_t, task_s, run_task_f, count_tasks_remaining
from flags cimport get_input, get_gradient, yield_output, yield_gradient
from flags cimport usleep


cdef struct args_s:
    flag_t* status
    int layer_id
    int N
    float* X
    float* Y
    float* dY
    float* dX
    int nr_dim
    int max_batch


cdef task_s make_task(flag_t* status, int layer_id,
        float* X, float* dX, int nr_dim, int batch_size, int N) nogil:
    args = <args_s*>calloc(1, sizeof(args_s))
    args.status = status
    args.layer_id = layer_id
    args.N = N
    args.X = X
    args.Y = X
    args.dY = dX
    args.dX = dX
    args.nr_dim = nr_dim
    args.max_batch = batch_size
    cdef task_s task
    task.run = <run_task_f>run_task
    task.args = args
    return task


cdef void* run_task(args_s* args) nogil:
    status = args.status
    layer_id = args.layer_id
    max_batch = args.max_batch
    N = args.N
    X = args.X
    Y = args.Y
    dX = args.dX
    dY = args.dY
    nr_dim = args.nr_dim
    cdef int fwd_todo, bwd_todo
    count_tasks_remaining(&fwd_todo, &bwd_todo, status, layer_id, N)
    if dY == NULL:
        bwd_todo = 0
    cdef int i, fwd_size, bwd_size
    while fwd_todo or bwd_todo:
        fwd_size = 0
        bwd_size = 0
        get_input(&i, &fwd_size, status, layer_id, N, N)
        for j in range(i, i+(nr_dim * fwd_size)):
            if X[j] < 0:
                Y[j] = 0
        yield_output(&status[i], fwd_size, layer_id)
        if fwd_todo < bwd_todo:
            get_gradient(&i, &bwd_size, status, layer_id, N, N)
            bwd_todo -= bwd_size
            memcpy(&dX[i], &dY[i], bwd_size * nr_dim * sizeof(float))
            for j in range(i, i+(nr_dim * bwd_size)):
                if X[j] < 0:
                    dX[j] = 0
            yield_gradient(&status[i], bwd_size, layer_id)
        if fwd_size == 0 and bwd_size == 0:
            break
            #usleep(100000) # Sleep for 0.1 seconds if no tasks were ready.
