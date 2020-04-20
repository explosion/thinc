import cupy.cudnn
import cupy.cuda.cudnn
from chainer.functions.rnn.n_step_lstm import n_step_lstm
from chainer.links.rnn.n_step_lstm import NStepLSTM


def use_chainer_lstm(num_layers, hidden_size, in_size, batch_size, seq_length):
    from chainer import Variable
    with cupy.cuda.Device(0):
        model = NStepLSTM(num_layers, in_size, hidden_size, dropout=0.0)
        model.to_gpu(0)
        xs = [
            Variable(cupy.random.rand(seq_length, in_size).astype(cupy.float32))
            for i in range(batch_size)
        ]
        hy, cy, ys = model(None, None, xs)
        print("Yay!")
    return ys
 


def cudnn_lstm_weight_concat(num_layers, states, ws, bs):
    n_W = 8
    out_size = ws[0].shape[0]
    in_size = ws[0].shape[1]
    dtype = ws[0].dtype
    cudnn_data_type = cupy.cudnn.get_data_type(dtype)


    rnn_desc = cupy.cudnn.create_rnn_descriptor(
        hidden_size=out_size,
        num_layers=num_layers,
        dropout_desc=states._desc,
        input_mode=cupy.cuda.cudnn.CUDNN_LINEAR_INPUT,
        direction=cupy.cuda.cudnn.CUDNN_UNIDIRECTIONAL,
        mode=cupy.cuda.cudnn.CUDNN_LSTM,
        data_type=cupy.cuda.cudnn.CUDNN_DATA_FLOAT,
        algo=None
    )

    dummy_x = cupy.zeros((1, in_size, 1), dtype="f")
    x_desc = cupy.cudnn.create_tensor_nd_descriptor(dummy_x)
    handle = cupy.cudnn.get_handle()
    cudnn_data_type = cupy.cudnn.get_data_type(dummy_x.dtype)
    assert cudnn_data_type == 0
    dtype = dummy_x.dtype
    byte_size = dtype.itemsize
    weights_size = cupy.cuda.cudnn.getRNNParamsSize( 
        handle, rnn_desc.value, x_desc.value, cudnn_data_type)
    w = cupy.zeros((weights_size // byte_size, 1, 1), dtype=dtype)
    w_desc = cupy.cudnn.create_filter_descriptor(w)

    for layer in range(num_layers): 
        for di in range(1): # Unidirectional 
            mat_index = layer * 1 + di 
            for lin_layer_id in range(n_W): # nW for LSTM 
                mat = cupy.cudnn.get_rnn_lin_layer_matrix_params(handle, rnn_desc, mat_index, x_desc, w_desc, w, lin_layer_id) 
                W_index = mat_index * n_W + lin_layer_id 
                m = mat.reshape(mat.size) 
                m[...] = ws[W_index].ravel()
                bias = cupy.cudnn.get_rnn_lin_layer_bias_params(handle, rnn_desc, mat_index, x_desc, w_desc, w, lin_layer_id) 
                b = bias.reshape(bias.size) 
                b[...] = bs[W_index] 
    return w

def alloc_params(num_layers, hidden_size, in_size):
    ws = []
    bs = []
    for i in range(num_layers):
        for j in range(8):
            # For first layer, we have four input-to-hidden weights. 
            in_dim = in_size if i == 0 and j < 4 else hidden_size
            ws.append(cupy.zeros((hidden_size, in_dim), dtype="f"))
            bs.append(cupy.zeros((hidden_size,), dtype="f"))
    return ws, bs


def recurrent_lstm_forward(Wb, Hx, Cx, X, lengths, is_train, *, bi, dropout_state):
    if is_train:
        reserve_space, Hy, Cy, Y = cupy.cudnn.rnn_forward_training(
            dropout_state,
            CUDNN_UNIDIRECTIONAL if not bi else CUDNN_BIDIRECTIONAL,
            CUDNN_LSTM,
            Hx,
            Cx,
            Wb,
            X,
            lengths
        )
        return Y, (reserve_space, Hx, Cx)
    else: 
        Hy, Cy, Y = cupy.cudnn.rnn_forward_inference(
            dropout_state,
            CUDNN_UNIDIRECTIONAL if not bi else CUDNN_BIDIRECTIONAL,
            CUDNN_LSTM,
            hx=Hx,
            cx=Cx,
            w=Wb,
            xs=X,
            lengths=lengths
        )
        null = xp.zeros((0, 0, 0), dtype="f")
        return Y, (null, null, null)


def main(num_layers=1, hidden_size=4, batch_size=3, in_size=10, seq_length=2):
    use_chainer_lstm(num_layers, hidden_size, in_size, batch_size, seq_length)
    states = cupy.cudnn.DropoutStates(None, 0)
    ws, bs = alloc_params(num_layers, hidden_size, in_size)
    w = cudnn_lstm_weight_concat(num_layers, states, ws, bs)

    # See 'check_type_forward' in chainer/functions/rnn/n_step_rnn/BaseNStepRNN
    hx = cupy.zeros((num_layers, batch_size, hidden_size), dtype="float32")
    cx = cupy.zeros((num_layers, batch_size, hidden_size), dtype="float32")

    xs = cupy.zeros((seq_length * batch_size, in_size), dtype="float32")
    lengths = [batch_size for _ in  range(seq_length)]

    print("My shapes")
    print("hx", hx.shape)
    print("cx", cx.shape)
    print("xs", xs.shape)
    print("w",w.shape)
    print("Lengths", lengths)
    reserve_space, hy, cy, ys = cupy.cudnn.rnn_forward_training(
        states,
        cupy.cuda.cudnn.CUDNN_UNIDIRECTIONAL,
        cupy.cuda.cudnn.CUDNN_LSTM,
        hx,
        cx,
        w,
        xs,
        lengths
    )

    dhy = hy
    dcy = cy
    dys = ys
    dhx, dcx, dxs = cupy.cudnn.rnn_backward_data(
        states,
        cupy.cuda.cudnn.CUDNN_UNIDIRECTIONAL,
        cupy.cuda.cudnn.CUDNN_LSTM,
        hx,
        cx,
        w,
        xs,
        ys,
        reserve_space,
        dhy,
        dcy,
        dys,
        lengths
    )

    dw = cupy.cudnn.rnn_backward_weights(
        states,
        cupy.cuda.cudnn.CUDNN_UNIDIRECTIONAL,
        cupy.cuda.cudnn.CUDNN_LSTM,
        xs,
        hx,
        ys,
        w,
        reserve_space,
        lengths
    )

    #hy, cy, ys = cupy.cudnn.rnn_forward_inference(
    #    states=states,
    #    direction_mode=cupy.cuda.cudnn.CUDNN_UNIDIRECTIONAL,
    #    rnn_mode=cupy.cuda.cudnn.CUDNN_LSTM,
    #    hx=hx,
    #    cx=cx,
    #    w=w,
    #    xs=xs,
    #    lengths=lengths
    #)

if __name__ == "__main__":
    main()
