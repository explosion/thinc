# cython: infer_types=True
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from linear cimport make_task as make_linear_task
from relu cimport make_task as make_relu_task
from optimizer cimport make_task as make_optimizer_task
from flags cimport flag_t, task_s, run_task_f
from params cimport params_s, init_params, free_params

import numpy
cimport numpy as np


cdef struct output_args_s:
    flag_t* status
    int layer_id
    float* guesses
    float* answers
    float* gradients
    int nr_out
    int max_batch
    int N
 

cdef task_s make_output_task(flag_t* status, int layer_id, 
        float* guesses, float* answers, float* gradients,
        int nr_out, int batch_size, int N):
    args = <output_args_s*>calloc(1, sizeof(output_args_s))
    args.status = status
    args.layer_id = layer_id
    args.guesses = guesses
    args.answers = answers
    args.gradients = gradients
    args.nr_out = nr_out
    args.max_batch = batch_size
    args.N = N
    cdef task_s task
    task.run = <run_task_f>run_output_task
    task.args = args
    return task


cdef void* run_output_task(output_args_s* args) nogil:
    for i in range(args.N):
        if args.status[i] == args.layer_id:
            idx = i * args.nr_out
            for j in range(idx, idx+args.nr_out):
                args.gradients[j] = args.guesses[j] - args.answers[j]
                args.status[j] = -args.layer_id
    

cdef struct input_args_s:
    flag_t* status
    int layer_id
    float* inputs
    float* queue
    int nr_dim
    int max_batch
    int N
 

cdef task_s make_input_task(flag_t* status, int layer_id, 
        float* inputs, float* queue,
        int nr_dim, int max_batch, int N):
    args = <input_args_s*>calloc(1, sizeof(input_args_s))
    args.status = status
    args.layer_id = layer_id
    args.inputs = inputs
    args.queue = queue
    args.nr_dim = nr_dim
    args.max_batch = max_batch
    args.N = N
    cdef task_s task
    task.run = <run_task_f>run_input_task
    task.args = args
    return task
 

cdef void* run_input_task(input_args_s* args) nogil:
    memcpy(args.queue, args.inputs, args.N*args.nr_dim*sizeof(float))
    for i in range(args.N):
        args.status[i] = args.layer_id+1


cdef launch_network(float* Xs, float* ys,
        int nO, int nI, int nH, int nB, int N):
    cdef int[3] sizes
    sizes[0] = nI * nH + nH
    sizes[1] = sizes[0] + nH*nH+nH
    sizes[2] = sizes[1] + nH*nO+nO
    params = <params_s*>calloc(3, sizeof(params_s))
    for i in range(3):
        params[i] = init_params(sizes[i])

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

    nr_task = 10
    tasks = <task_s*>calloc(nr_task, sizeof(task_s))
    tasks[0] = make_input_task(status, 1, Xs, inputs, nI, N, N)
    tasks[1] = make_linear_task(status, 3, &params[0],
                                inputs, hidden1, d_hidden1, NULL,
                                nH, nI, nB, N)
    tasks[2] = make_relu_task(status, 5, hidden1, d_hidden1, nH, N, N)
    tasks[3] = make_linear_task(status, 7, &params[1],
                                hidden1, hidden2, d_hidden2, d_hidden1,
                                nH, nH, nB, N)
    tasks[4] = make_relu_task(status, 9, hidden2, d_hidden2, nH, N, N)
    tasks[5] = make_linear_task(status, 11, &params[2],
                                hidden2, outputs, d_outputs, d_hidden2,
                                nH, nO, nB, N)
    tasks[6] = make_output_task(status, 12, outputs, ys, d_outputs, nO, N, N)
    tasks[7] = make_optimizer_task(&params[0], 0.01)
    tasks[8] = make_optimizer_task(&params[1], 0.01)
    tasks[9] = make_optimizer_task(&params[2], 0.01)
 
    for i in range(2):
        print("Status before", status[0])
        tasks[i].run(tasks[i].args)
    #for i in range(nr_task):
    #    free(tasks[i].args)
    #free(tasks)


def main(int N=10, nI=4, nO=8, nH=3, nB=2):
    cdef np.ndarray Xs = numpy.ones((N, nI), dtype='f')
    cdef np.ndarray ys = numpy.ones((N, nO), dtype='f')
    launch_network(<float*>Xs.data, <float*>ys.data,
        nO, nI, nH, nB, N)

 
if __name__ == '__main__':
    main()
