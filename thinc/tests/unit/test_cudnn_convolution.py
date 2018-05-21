import pytest
from ...neural.ops import CupyOps

def test_cudnn_convolution_forward():
    ops = CupyOps()
    X = ops.allocate((6, 4))
    W = ops.allocate((4*3, 4*3))
    b = ops.allocate((4*3,))
    Y = ops.cudnn_convolution(X, W, b)
    assert Y.shape == (6, 4*3)
