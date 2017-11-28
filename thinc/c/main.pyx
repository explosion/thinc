include "linear_layer.pyx"

cdef task_s make_output_task(flag_t* status, int layer_id, 
        float* guesses, float* answers, float* gradients,
        int nr_out, int batch_size, int N):
    args = <output_args_s*>calloc(1, sizeof(output_args_s))
    args.status = status
    args.layer_id = layer_id
    args.guesses = guesses
    args.answers = answers
    args.gradients = gradients
    args.nr_out = nr_dim
    args.max_batch = max_batch
    args.N = N
    return task_s(run=output_layer, args=args)
    

cdef task_s make_input_task(flag_t* status, int layer_id, 
        float* inputs, float* queue,
        int nr_out, int max_batch, int N):
    args = <output_args_s*>calloc(1, sizeof(input_args_s))
    args.status = status
    args.layer_id = layer_id
    args.inputs = inputs
    args.queue = queue
    args.nr_out = nr_dim
    args.max_batch = max_batch
    args.N = N
    return task_s(run=set_input, args=args)
 
cdef launch_network(float* Xs, float* ys,
        int nO, int nI, int nH, int nB, int N):
    cdef int[3] sizes
    sizes[0] = nI * nH + nH
    sizes[1] = sizes[0] + nH*nH+nH
    sizes[2] = sizes[1] + nH*nO+nO
    nr_param = sizes[2]

    all_params = <float*>calloc(nr_param, sizeof(float))
    all_d_params = <float*>calloc(nr_param, sizeof(float))
    params = <float***>calloc(3, sizeof(float*))
    for i in range(3):
        params[i] = <float**>calloc(2, sizeof(float*))
        params[i][0] = &all_params[sizes[i]]
        params[i][1] = &all_d_params[sizes[i]]

    status = <flag_t*>calloc(N, sizeof(flag_t)) 
    # Activations
    inputs = <float*>calloc(nI*N, sizeof(float))
    hidden1 = <float*>calloc(nH*N, sizeof(float))
    hidden2 = <float*>calloc(nH*N, sizeof(float))
    outputs = <float*>calloc(nO*N, sizeof(float))
    # Gradients
    d_inputs = NULL
    d_hidden1 = <float*>calloc(nH*N, sizeof(float))
    d_hidden2 = <float*>calloc(nH*N, sizeof(float))
    d_outputs = <float*>calloc(nO*N, sizeof(float))

    nr_task = 8
    tasks = <task_s*>calloc(nr_task, sizeof(task_s))
    tasks[0] = Inputs.make_task(status, 1, Xs, inputs, nr_in, N)
    tasks[1] = Linear.make_task(status, 3, &params[0],
                                inputs, hidden1, d_hidden1, NULL,
                                nH, nI, batch_size, N)
    tasks[2] = ReLu.make_task(status, 5, hidden1, d_hidden1, nH, N, N)
    tasks[3] = Linear.make_task(status, 7, &params[1],
                                hidden1, hidden2, d_hidden2, d_hidden1,
                                nH, nH, batch_size, N)
    tasks[4] = ReLu.make_task(status, 9, hidden2, d_hidden2, nH, N, N)
    tasks[5] = Linear.make_task(status, 11, &params[2],
                                hidden2, output, d_output, d_hidden2,
                                nH, nO, batch_size, N)
    tasks[6] = Output.make_task(status, 13, output, d_output, nO, N, N)
    tasks[7] = Optimizer.make_task(params, gradients, nr_upd, nr_param)
 
    for i in range(10):
        for i in range(nr_task):
            tasks[i].run(&ops[i].args)
    for i in range(nr_task):
        free(tasks[i].args)
    free(tasks)
    free(all_param)
    free(all_d_params)
    for i in range(3):
        free(params[i])
    free(params)

