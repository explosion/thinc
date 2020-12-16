cdef void check_status(status_t status) nogil:
    pass


cdef void* create_rnn_descriptor() nogil:
    cdef size_t address = 0
    check_status(cudnnCreateRNNDescriptor(<cudnnRNNDescriptor_t*>&address))
    return <void*>address


cdef void destroy_rnn_descriptor(void* address) nogil:
    check_status(cudnnDestroyRNNDescriptor(<cudnnRNNDescriptor_t>address)


cdef void* create_rnn_data_descriptor() nogil:
    cdef size_t address = 0
    check_status(cudnnCreateRNNDataDescriptor(<cudnnRNNDataDescriptor_t*>&address))
    return <void*>address


cdef void destroy_rnn_data_descriptor(void* address) nogil:
    check_status(cudnnDestroyRNNDataDescriptor(<cudnnRNNDataDescriptor_t>address)


cdef void set_bilstm_descriptor(
	void* rnn_desc,
	cudnnDataType_t data_type,
	cudnnDataType_t math_prec,
	cudnnMathType_t math_type,
	int32_t input_size,
	int32_t hidden_size,
	int32_t proj_size,
	int32_t num_layers,
    void* dropout_desc,
    uint32_t aux_flags
) nogil:
    check_status(cudnnSetRNNDescriptor_v8(
        <cudnnRNNDescriptor_t>rnn_desc
        CUDNN_RNN_ALGO_STANDARD,
        CUDNN_RNN_CELL_MODE_LSTM,
        CUDNN_RNN_SINGLE_INP_BIAS,
        CUDNN_BIDIRECTIONAL,
        CUDNN_LINEAR_INPUT,
        data_type,
        math_prec,
        math_type,
        input_size,
        hidden_size,
        proj_size,
        num_layers,
        <cudnnDropoutDescriptor_t>dropout_desc,
        aux_flags
    ))


cdef void set_bilstm_data_descriptor(
    void* address,
    size_t data_type,
    int max_seq_length,
    int batch_size,
    int vector_size,
    const int* seq_length_array,
    const int* padding_fill
) nogil:
    check_status(cudnnSetRNNDataDescriptor(
        <cudnnRNNDataDescriptor_t>address,
        <cudnnDataType_t>data_type,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        max_seq_length,
        batch_size,
        vector_size,
        const int* seq_length_array,
        padding_fill
    ))


cdef void set_seq_data_descriptor(
    void* seq_data_desc,
    cudnnDataType_t data_type,
	int nb_dims,
	const int* dim_A,
	const void* axes,
	size_t seq_length_array_stride
	const int* seq_length_array,
	void* padding_fill
) nogil:
    check_status(cudnnSetSeqDataDescriptor(
        <cudnnSeqDataDescriptor_t>seqDataDesc,
        data_type,
        nb_dims,
        dim_A,
        <const cudnnSeqDataAxis_t*>axes,
        seq_length_array_stride,
        seq_length_array,
        padding_fill
    ))


cdef void bilstm_forward_inference(
    const void* handle,
    const void* rnn_desc,
    const int32_t* dev_seq_lengths,
    const void* x_desc,
    const void* x,
    const void* y_desc,
    void* y,
    const void* h_desc,
    const void *hx,
    void *hy,
    const void* c_desc,
    const void *cx,
    void *cy,
    size_t n_weight_bytes,
    const void weights,
    size_t n_workspace_bytes,
    void* workspace,
    size_t n_reserve_bytes,
    void* reserve
) nogil:
    check_status(cudnnRNNForward(
        handle,
        (<cudnnRNNDescriptor_t*>rnn_desc)[0],
        CUDNN_FWD_MODE_INFERENCE,
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], x,
        (<cudnnRNNDataDescriptor_t*>yDesc)[0], y,
        (<cudnnRNNDataDescriptor_t*>h_desc)[0], hx, hy,
        (<cudnnRNNDataDescriptor_t*>)c_desc)[0], cx, cy,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve_space
    ))


cdef void bilstm_forward_training(
    const void* handle,
    const void* rnn_desc,
    const int32_t* dev_seq_lengths,
 
) nogil:
    check_status(cudnnRNNForward(
        handle,
        (<cudnnRNNDescriptor_t*>rnn_desc)[0],
        CUDNN_FWD_MODE_TRAINING,
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], x,
        (<cudnnRNNDataDescriptor_t*>yDesc)[0], y,
        (<cudnnRNNDataDescriptor_t*>h_desc)[0], hx, hy,
        (<cudnnRNNDataDescriptor_t*>)c_desc)[0], cx, cy,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve_space
    ))


cdef void bilstm_backward(
    const void* handle,
    const void* rnn_desc,
    const int32_t* dev_seq_lengths,
    size_t y_desc,
    const void *y,
    const void *dy,
    size_t x_desc,
    void *dx,
    const void* h_desc,
    const void *hx,
    const void *dhy,
    void* dhx,
    const void* c_desc,
    const void *cx,
    const void *dcy,
    void *dcx,
    size_t n_weight_bytes,
    const void *weights,
    size_t n_workspace_bytes,
    void* workspace,
    size_t n_reserve_bytes,
    void* reserve
) nogil:
    check_status(cudnnRNNBackwardData_v8(
        (<cudnnHandle_t*>)handle)[0],
        (<cudnnRNNDescriptor_t*>rnnDesc)[0],
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>y_desc)[0], y, dy,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], dx,
        (<cudnnTensorDescriptor_t*>h_desc)[0], hx, dhy, dhx,
        (<cudnnTensorDescriptor_t*>c_desc)[0], cx, dcy, dcx,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ));
    check_status(cudnnRNNBackwardData_v8(
        (<cudnnHandle_t*>)handle)[0],
        (<cudnnRNNDescriptor_t*>rnnDesc)[0],
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>y_desc)[0], y, dy,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], x, dx,
        (<cudnnTensorDescriptor_t*>h_desc)[0], dx, dhy, dhx,
        (<cudnnTensorDescriptor_t*>cDesc)[0], cx, dcy, dcx,
        weightSpaceSize, weightSpace,
        workSpaceSize, workSpace,
        reserveSpaceSize, reserveSpace
    ))


cdef void multi_head_attn_forward_inference(
    size_t handle,
    size_t attn_desc,
    int curr_idx,
    const int* lo_win_idx,
	const int* hi_win_idx,
	const int* dev_seq_lengths_QO,
	const int* dev_seq_lengths_KV,
	size_t q_desc,
	const void* queries,
	const void* residuals,
	size_t k_desc,
	const void* keys,
	size_t v_desc,
	const void* values,
	size_t o_desc,
    void *out,
	size_t n_weight_bytes,
	const void* weights,
	size_t n_workspace_bytes,
	void* workspace
) nogil:
    check_status(cudnnMultiHeadAttnForward(
	    <cudnnHandle_t>handle,
	    <const cudnnAttnDescriptor_t>attn_desc,
	    curr_idx,
        lo_win_idx,
        hi_win_idx,
        dev_seq_lengths_QO,
        dev_Seq_lengths_KV,
	    <const cudnnSeqDataDescriptor_t>q_desc, queries, residuals,
	    <const cudnnSeqDataDescriptor_t>k_desc, keys,
	    <const cudnnSeqDataDescriptor_t>v_desc, values,
	    <const cudnnSeqDataDescriptor_t>o_desc, out,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        0, NULL
    ))

cdef void multi_head_attn_forward_training(
    size_t handle,
    size_t attn_desc,
    int curr_idx,
    const int* lo_win_idx,
	const int* hi_win_idx,
	const int* dev_seq_lengths_QO,
	const int* dev_seq_lengths_KV,
	size_t q_desc,
	const void* queries,
	const void* residuals,
	size_t k_desc,
	const void* keys,
	size_t v_desc,
	const void* values,
	size_t o_desc,
    void *out,
	size_t n_weight_bytes,
	const void* weights,
	size_t n_workspace_bytes,
	void* workspace,
    size_t n_reserve_bytes,
	void* reserve
) nogil:
    check_status(cudnnMultiHeadAttnForward(
	    <cudnnHandle_t>handle,
	    <const cudnnAttnDescriptor_t>attn_desc,
	    curr_idx,
        lo_win_idx,
        hi_win_idx,
        dev_seq_lengths_QO,
        dev_Seq_lengths_KV,
	    <const cudnnSeqDataDescriptor_t>q_desc, queries, residuals,
	    <const cudnnSeqDataDescriptor_t>k_desc, keys,
	    <const cudnnSeqDataDescriptor_t>v_desc, values,
	    <const cudnnSeqDataDescriptor_t>o_desc, out,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ))


cdef void multi_head_attn_backward(
    size_t handle,
    size_t attn_desc,
    const int* lo_win_idx,
    const int* hi_win_idx,
    const int* dev_seq_lengths_DQDO,
    const int* dev_seq_lengths_DQDV,
    size_t do_desc,
    void* dout,
    size_t dq_desc,
    const void* queries,
    size_t dk_desc,
    void* dkeys,
    const void* keys,
    size_t dv_desc,
    void* dvalues,
    const void* values,
    size_t n_weight_bytes,
    void* dweights,
    const void* weights,
    size_t n_workspace_bytes,
    void* workspace,
    size_t n_reserve_bytes,
    void* reserve
) nogil:
    check_status(cudnnMultiHeadAttnBackwardData(
    	<cudnnHandle_t>handle,
	    <const cudnnAttnDescriptor_t>attn_desc,
        lo_win_idx,
        hi_win_idx,
        dev_seq_lengths_DQDO,
        dev_seq_lengths_DKDV,
        <const cudnnSeqDataDescriptor_t>do_desc, dout,
	    <const cudnnSeqDataDescriptor_t>dq_desc, dqueries, queries,
	    <const cudnnSeqDataDescriptor_t>dk_desc, dkeys, keys,
	    <const cudnnSeqDataDescriptor_t>dv_desc, dvalues, values,
	    n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ))
    check_status(cudnnMultiHeadAttnBackwardWeights(
    	<cudnnHandle_t>handle,
	    <const cudnnAttnDescriptor_t>attn_desc,
        <cudnnWgradMode_t>addGrad,
	    <const cudnnSeqDataDescriptor_t>q_desc, queries,
	    <const cudnnSeqDataDescriptor_t>k_desc, keys,
	    <const cudnnSeqDataDescriptor_t>v_desc, values,
	    <const cudnnSeqDataDescriptor_t>do_desc, dout,
	    n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ))
