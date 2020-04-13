from dataclasses import dataclass

try:
    import cupy
    from cupy.cuda import cudnn
except ImportError:
    pass

# Weights format for RNN
# Taken me ages to try to piece this together...
# Created via
# 
# int dimW[3];
# dimW[0] = weightsSize / sizeof(T_ELEM);
# dimW[1] = 1;
# dimW[2] = 1;

# cudnnErrCheck(cudnnSetFilterNdDescriptor(wDesc, getDataType<T_ELEM>(), CUDNN_TENSOR_NCHW, 3, dimW));
#
# Docs:
# https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnSetFilterNdDescriptor
# 
# The docs say that when n=3, 
# K represents the number of output feature maps
# C is the number of input feature maps
# R is the number of rows per filter
# 
# So it creates it as a block in the "K" dimension...Which doesn't reveal the
# actual structure. Bugger.

class Output:
    pass

class PointerArray:

    def __init__(self, lst, back_pointer):
        self._value = numpy.array(lst, dtype=numpy.intp)
        # Store back_pointer to prevent the GC removes the original variable
        self._back_pointer = back_pointer

    @property
    def data(self):
        return self._value.ctypes.data


def _make_tensor_descriptor_array(xs):
    """Make an array of pointers denoting pointers of tensor descriptors.
    """
    descs = []
    for x in xs:
        if x.ndim < 3:
            shape = x.shape + (1,) * (3 - x.ndim)
            x = x.reshape(shape)
        desc = cudnn.create_tensor_nd_descriptor(x)
        descs.append(desc)
    return PointerArray([d.value for d in descs], descs)


def _make_ptr_array(xs):
    """Make an array of pointers denoting pointers of ndarrays.
    """
    return PointerArray([x.data.ptr for x in xs], xs)


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]


@dataclass
class CuDNNTensor:
    desc: Pointer
    data: Pointer
    array: Array

    def __init__(self, array: Array):
        self.array = array


@dataclass
class LSTM:
    W: Array2d,
    b: Array1d,
    h_init: Array1d,
    c_init: Array1d,


def cudnn_rnn_forward(
    inputs: Tuple[Array3d, Array3d, Array3d],
    params: Tuple[Array3d, Array3d, Array3d],
    is_train: bool,
    bi: bool=False,
    activation: str="lstm",
    dropout: float=0.0
) -> Tuple[Array3d, Tuple[Array3d, Array3d, Array3d]]:
    """
    Params: (W, b, init_h, init_c)
        W (Array3d): Weights, shaped (layer, 
    dropout_state: DropoutStates states
    direction_mode = CUDN_BIDIRECTIONAL if bi else CUDNN_UNIDIRECTIONAL
    rnn_mode = CUDNN_RNN_ACTIVATION_TYPES[activation]

    reserve_space, H, C, Y = cupy.cudnn.rnn_forward_training(
        states,
        direction_mode,
        rnn_mode,
        H,
        C,
        Wb,
        X,
        lengths
    )
