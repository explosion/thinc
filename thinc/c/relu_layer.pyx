from libc.string cimport memcpy


cdef struct relu_args_s:
    flags_t* status
    int layer_id
    int N
    const float* X
    float* Y
    const float* dY
    float* dX
    int nr_dim
    int max_batch


cdef void relu_layer(relu_args_s* args) nogil:
    status = args.status
    layer_id = args.layer_id
    N = args.N
    X = args.X
    Y = args.Y
    dX = args.dX
    dY = args.dY
    cdef int i, size
    while tasks_remaining(status, layer_id, N):
        await_input(&i, &size, status, N, max_batch, layer_id)
        for j in range(i, i+(nr_dim * size)):
            if X[j] < 0:
                Y[j] = 0
        yield_output(&status[i], size, layer_id)
        if dY != NULL and dX != NULL:
            await_gradient(&status[i], size, layer_id)
            memcpy(dX, dY, size * sizeof(float))
            for j in range(i, i+(nr_dim * size)):
                if X[j] < 0:
                    dX[j] = 0
            yield_gradient(&status[i], size, layer_id)
