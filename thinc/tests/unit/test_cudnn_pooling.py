from ...neural._cudnn_pooling import max_pool
from ...neural.ops import CupyOps


def test_max_pool_forward():
    ops = CupyOps()
    # Batch 2, ndim 2
    X = ops.allocate((2, 2))
    X[0,0] = 1.
    X[1,1] = 1.
    Y, get_dX = max_pool(X)
    Y = Y.get()
    assert Y[0] == 1.
    assert Y[1] == 1.
