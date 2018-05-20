# cython: infer_types=True
from libcpp.vector cimport vector

from cupy.cuda import cudnn as libcudnn
from cupy import cudnn as py_cudnn
from cupy import core


cdef class PoolingDescriptor:
    cdef size_t c

    def __init__(self, mode, int window_height, int window_width,
            int v_pad, int h_pad, int v_stride, int h_stride):
        self.c = libcudnn.createPoolingDescriptor()
        libcudnn.setPooling2dDescriptor_v4(self.c,
            mode, libcudnn.CUDNN_NOT_PROPAGATE_NAN,
            window_height, window_width, v_pad, h_pad, v_stride, h_stride)

    def __del__(self):
        libcudnn.destroyPoolDescriptor(self.c)


cdef class TensorDescriptor:
    cdef size_t c
    
    def __init__(self, arr, format):
        if not arr.flags.c_contiguous:
            raise ValueError('cupy.cudnn supports c-contiguous arrays only')
        self.c = libcudnn.createTensorDescriptor()
        if arr._shape.size() == 4:
            libcudnn.setTensor4dDescriptor(self.c, format,
                py_cudnn.get_data_type(arr.dtype),
                arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3])
        else:
            _set_tensor_nd_descriptor(self.c, py_cudnn.get_data_type(self.c), arr)

    def __del__(self):
        libcudnn.destroyTensorDescriptor(self.c)


def _set_tensor_nd_descriptor(size_t desc, int data_type, arr):
    cdef vector[int] stride_in_elems
    cdef vector[int] shape
    cdef int itemsize = arr.itemsize
    cdef int ndim = arr.ndim
    for s in range(ndim):
        stride_in_elems.push_back(arr.stride[s] // itemsize)
        shape.push_back(arr.shape[s])
    cdef size_t shape_ptr = <size_t>&shape[0]
    cdef size_t stride_ptr = <size_t>&stride_in_elems[0]
    libcudnn.setTensorNdDescriptor(
        desc, data_type, ndim, shape_ptr, stride_ptr)
