void check_status(status_t status) {
};

void bilstm_forward_inference(

)
{
    check_status(cudnnRNNForward(
        handle,
        rnnDesc,
        CUDNN_FWD_MODE_INFERENCE,
        devSeqLengths,
        xDesc, x,
        yDesc, y,
        hDesc, hx, hy,
        cDesc, cx, cy,
        weightSpaceSize, weightSpace,
        workSpaceSize, workSpace,
        reserveSpaceSize, reserveSpace
    ));
}

void bilstm_forward_training(

)
{
    check_status(cudnnRNNForward(
        handle,
        rnnDesc,
        CUDNN_FWD_MODE_TRAINING,
        devSeqLengths,
        xDesc, x,
        yDesc, y,
        hDesc, hx, hy,
        cDesc, cx, cy,
        weightSpaceSize, weightSpace,
        workSpaceSize, workSpace,
        reserveSpaceSize, reserveSpace
    ));
}


void bilstm_backward(
    const void* handle,
    const void* rnn_desc,
    const int32_t* devSeqLengths,
    size_t y_desc,
    const void *y,
    const void *dy,
    size_t x_desc,
    void *dx,
    const void* h_desc,
    const void *hx,
    const void *dhy,
    void *dhx,
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
)
{
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
    ));
}


void multi_head_attn_forward_inference(
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
)
{
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
        0, NULL));
}

void multi_head_attn_forward_training(
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
	void* reserve);
{
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
        n_reserve_bytes, reserve));
}


void multi_head_attn_backward(
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
)
{
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
    ));
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
    ));
}
