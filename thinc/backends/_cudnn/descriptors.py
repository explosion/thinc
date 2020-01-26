#########################
# CuDNN Enums
# 
# CuDNN follows C idioms, so most args are enum values. Map these to strings
# to be more legible to Python.
#
#########################

RNN_MODES = {
    "relu": cudnn.CUDNN_RNN_RELU # 0
    "tanh": cudnn.CUDNN_RNN_TANH,  # l
    "lstm": cudnn.CUDNN_LSTM # 2
    "gru": cudnn.CUDNN_GRU # 3
}

RNN_ALGORITHMS = {
    "standard": cudnn.CUDNN_RNN_ALGO_PERSIST_STATIC,
    "persist_static": cudnn.CUDNN_RNN_ALGO_PERSIST_STATIC,
    "persist_dynamic": cudnn.CUDNN_RNN_ALGO_PERSIST_DYNAMIC
}

RNN_DATA_LAYOUTS = {
    "seq_major_unpacked": cudnn.CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
    "seq_major_packed": cudnn.CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
    "batch_major_unpacked": cudnn.CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
}

RNN_CLIP_MODE = {
    False: cudnn.CUDNN_RNN_CLIP_NONE,
    True: cudnn.CUDNN_RNN_CLIP_MINMAX
}

CROSS_CONVOLUTION = {
    False: cudnn.CUDNN_CONVOLUTION, # = 0
    True: cudnn.CUDNN_CROSS_CORRELATION #= 1
}

CONVOLUTION_FWD = {
    "no_workspace": cudnn.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, #= 0
    "prefer_fastest": cudnn.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, #= 1
    "specify_workspace_limit": cudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, #= 2
}

CONVOLUTION_FWD_ALGO = {
    "implicit_gemm": cudnn.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, #= 0
    "implicit_precomp_gemm": cudnn.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, #= 1
    "gemm": cudnn.CUDNN_CONVOLUTION_FWD_ALGO_GEMM, # = 2
    "direct": cudnn.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, # = 3
    "fft": cudnn.CUDNN_CONVOLUTION_FWD_ALGO_FFT, # = 4
    "fft_tiling": cudnn.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING, #= 5
    "winograd": cudnn.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, #= 6
    "winograd_nonfused": cudnn.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED #= 7
}

CONVOLUTION_BWD_FILTER = {
    "no_workspace": cudnn.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, # = 0
    "prefer_fastest": cudnn.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, # = 1
    "specify_limit": cudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT # = 2
}

CONVOLUTION_BWD_FILTER_ALGO = {
    1: cudnn.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, #= 0
    2: cudnn.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, #= 1
    "fft": cudnn.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT, #= 2
    3: cudnn.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3, #= 3
    "winograd": cudnn.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD, #= 4
    "winograd_nonfused": cudnn.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED #= 5
}

CONVOLUTION_BWD_DATA = {
    "no_workspace": cudnn.CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, # = 0
    "prefer_fastest": cudnn.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, # = 1
    "specify_limit": cudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT #= 2
}

CONVOLUTION_BWD_DATA_ALGO = {
    0: cudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, # = 0
    1: cudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, # = 1
    "fft": cudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, # = 2
    "tiling": cudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING, # = 3
    "winograd": cudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD, # = 4
    "winograd_nonfused": cudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED, #= 5
}


DATA_TYPES = {
    "float32":  cudnn.CUDNN_DATA_FLOAT,
    "float16": cudnn.CUDNN_DATA_HALF,
    "float64":  cudnn.CUDNN_DATA_DOUBLE,
}


@dataclass
class Descriptor:
    _ptr: int

    @property
    def ptr(self):
        if self._ptr == 0:
            self.create()
        return self._ptr
    
    def create(self):
        raise NotImplementedError

    def destroy(self):
        raise NotImplementedError

@dataclass
class ConvolutionDescriptor(Descriptor):
    _ptr: int
    
    def create(self):
        self._ptr = cudnn.createConvolutionDescriptor()
        if self.ndim == 2:
            cudnn.setConvolution2dDescriptor_v4(...)
        else:
            cudnn.setConvolutionNdDescriptor_v3(...)

    def destroy(self):
        cudnn.destroyConvolutionDescriptor(self._ptr)


@dataclass
class PoolingDescriptor(Descriptor):
    def create(self):
        self._ptr = cudnn.createPoolingDescriptor()
        if self.ndim == 2:
            cudnn.setPooling2dDescriptor_v4(...)
        else:
            cudnn.setPoolingNdDescriptor_v4(...)

    def destroy(self):
        cudnn.destroyPoolingDescriptor(self._ptr)


@dataclass
class DropoutDescriptor(Descriptor):
    prob: float
    size: int
    seed: int=0
    _ptr: int

    def create(self) -> None:
        self._ptr = cudnn.createDropoutDescriptor()
        cudnn.cudnnSetDropoutDescriptor(
            self._ptr,
            self.handle.ptr,
            self.prob,
            self.data.ptr,
            self.size,
            self.seed
        )

    def destroy(self) -> None:
        cudnn.destroyDropoutDescriptor(self._ptr)


@dataclass
class RNNDescriptor:
    handle: Handle
    n_unit: int
    n_layer: int
    dropout: DropoutDescriptor

    is_linear_input: bool
    is_unidirectional: bool
    activation_type: str
    data_type: str
    algorithm: str = "standard"
    _ptr: int

    def create(self):
        self._ptr = cudnn.createRNNDescriptor()
        cudnn.setRNNDescriptor_v6(
            self.n_init,
            self.n_layer,
            self.dropout.ptr,
            CUDNN_LINEAR_INPUT if self.is_linear_input else CUDNN_SKIP_INPUT,
            CUDNN_BIDIRECTIONAL if self.bi else CUDNN_UNIDIRECTIONAL,
            CUDNN_RNN_ACTIVATION_TYPES[self.activation_type],
            CUDNN_RNN_ALGO_STANDARD[self.algorithm],
            CUDNN_DATA_TYPES[self.data_type]
        )

    def destroy(self):
        self.dropout.destroy()
        cupy.cuda.cudnn.destroyRNNDataDescriptor(self._rnn_desc)
        self._ptr = 0


@dataclass
class RNNDataDescriptor:
    handle: Handle

    data_type: str
    layout: str
    max_length: int
    batch_size: int
    vector_size: int
    padding_fill: int`
    seq_length_array: int

    def create(self):
        self._ptr = cudnn.createRNNDataDescriptor()
        cudnn.setRNNDataDescriptor(
            self._ptr,
            dataType,
            layout,
            maxSeqLength,
            batchSize,
            vectorSize,
            seqLengthArray,
            paddingFill
        )

    def destroy(self):
        cudnn.destroyRNNDataDescriptor(self._ptr)


