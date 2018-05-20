from cupy.cuda import cudnn as libcudnn
from cupy import cudnn as cudnn
import cupy
import numpy
from . import _chainer_utils as _chainer

from ._cudnn_utils cimport TensorDescriptor, PoolingDescriptor


def _cudnn_pooling(pool_type, height=1):
    def cudnn_pooling_forward(X, drop=0.):
        X = cupy.ascontiguousarray(X)
        Y = cupy.empty((X.shape[0], X.shape[1], 1, 1), dtype=X.dtype)
        handle = cudnn.get_handle()
        pool_desc = PoolingDescriptor(mode, 1, X.shape[1], 0, 0, 1, X.shape[1])
        x_desc = TensorDescriptor(X)
        y_desc = TensorDescriptor(Y)
        _cudnn_pool_forward(Y.data.ptr,
            y_desc.c, X.data.ptr, x_desc.c,
            pool_desc.c, handle)
        def cudnn_pooling_backward(dY, sgd=None):
            handle = cudnn.get_handle()
            pool_desc = PoolingDescriptor(mode, X.shape[1], 1, 0, 0, 1, X.shape[1])
            x_desc = TensorDescriptor(X)
            dy_desc = TensorDescriptor(dY)

            dX = cupy.empty_like(X)
            _cudnn_pool_backward(dX.data.ptr,
                dx_desc.value,
                dY.data.ptr, dy.desc.value, Y.data.ptr, y_desc.value,
                X.data.ptr, x_desc.value, pool_desc.value, handle)
            return dX
        return Y, cudnn_pooling_backward
    return cudnn_pooling_forward


max_pool = _cudnn_pooling(libcudnn.CUDNN_POOLING_MAX)
mean_pool = _cudnn_pooling(libcudnn.CUDNN_POOLING_AVERAGE_COUNT_INCLUDING_PADDING)


def _cudnn_pool_forward(Y_data_ptr, y_desc_value, X_data_ptr, x_desc_value,
        pool_desc_value, handle):
    one = numpy.array(1, dtype=oz_dtype).ctypes
    zero = numpy.array(0, dtype=oz_dtype).ctypes
    libcudnn.poolingForward(
        handle, pool_desc.value, one.data, x_desc.value,
        x.data.ptr, zero.data, y_desc.value, y.data.ptr)


def _cudnn_pool_backward():
    libcudnn.poolingBackward(
        handle, pool_desc.value, one.data, y_desc.value,
        Y.data.ptr, y_desc.value, dY.data.ptr, x_desc.value,
        X.data.ptr, zero.data, x_desc.value, dX.data.ptr)
