import pytest
import time
from numpy.testing import assert_allclose
import numpy.random
from timeit import default_timer as timer
from ...neural._classes.maxout import Maxout

from ...neural._fast_maxout_cnn import MaxoutWindowEncoder

numpy.random.seed(0)

def test_create():
    mwe = MaxoutWindowEncoder(16, 4)


def test_fwd_runs():
    mwe = MaxoutWindowEncoder(32, 4)
    X = mwe.ops.allocate((5, 32), dtype='f')
    y = mwe([X])[0]
    assert y.shape == X.shape
    assert y.sum() == 0.
    y += 2
    z = mwe([y])[0]
    assert z.sum() != 0.


def test_bwd_runs():
    mwe = MaxoutWindowEncoder(32, 4)
    X = mwe.ops.allocate((5, 32), dtype='f')
    dy = mwe.ops.allocate((5, 32), dtype='f')
    y, bp_y = mwe.begin_update([X])
    dX = bp_y([dy])


def baseline_mwe(nO, nP, depth):
    from thinc.neural._classes.model import Model
    from thinc.neural._classes.resnet import Residual
    from thinc.neural._classes.convolution import ExtractWindow
    from thinc.neural._classes.layernorm import LayerNorm
    from thinc.api import chain, clone, with_flatten
    maxout = Maxout(nO, nO*3, pieces=nP)
    normalize = LayerNorm(maxout)
    with Model.define_operators({'>>': chain, '**': clone}):
        model = Residual(ExtractWindow(nW=1) >> normalize)
        model = with_flatten(chain(*([model]*depth)))
    model.maxout = maxout
    model.normalize = normalize
    return model


def test_fwd_correctness(nr_row=20, nr_dim=5, nr_piece=3):
    base = baseline_mwe(nr_dim, 3, 4)
    fast = MaxoutWindowEncoder(nr_dim, 4)
    fast.maxout.W[:] = base.maxout.W
    fast.normalize.G[:] = base.normalize.G
    Xs = [fast.ops.normal_init(fast.ops.allocate((nr_row, nr_dim)), nr_dim)
          for _ in range(10)]
    Ys_new = fast(Xs)
    # Because of the flattening, this isn't correct if just base(Xs)
    Ys_old = [base([X])[0] for X in Xs]
    for Y1, Y2 in zip(Ys_new, Ys_old):
        assert_allclose(Y1, Y2, rtol=0.0001, atol=0.0001)

@pytest.mark.xfail
def test_bwd_correctness(nr_row=2, nr_dim=2, nr_piece=3):
    base = baseline_mwe(nr_dim, 3, 2)
    fast = MaxoutWindowEncoder(nr_dim, 2)
    fast.maxout.W[:] = base.maxout.W
    fast.normalize.G[:] = base.normalize.G
    Xs = [fast.ops.normal_init(fast.ops.allocate((nr_row, nr_dim)), nr_dim)
          for _ in range(3)]
    Ys, bp_Ys = fast.begin_update(Xs)
    dXs_new = bp_Ys(Xs)
    dXs_old = []
    for X in Xs:
        Y, bp_Y = base.begin_update([X])
        dXs_old.append(bp_Y([X])[0])
    for dX1, dX2 in zip(dXs_new, dXs_old):
        assert_allclose(dX1, dX2, rtol=1e-2, atol=1e-2)


def test_fwd_speed(nr_row=100, nr_dim=128, nr_piece=3):
    mwe = MaxoutWindowEncoder(nr_dim, 4)
    Xs = [mwe.ops.allocate((nr_row, nr_dim)) for _ in range(100)]
    start = timer()
    ys = mwe(Xs)
    end = timer()
    print('Fwd Fast?', end, start, end-start)
    mwe = baseline_mwe(nr_dim, nr_piece, 4)
    start = timer()
    y = mwe(Xs)
    end = timer()
    print('Fwd Slow?', end, start, end-start)


def test_bwd_speed(nr_row=30, nr_dim=128, nr_piece=3):
    mwe = MaxoutWindowEncoder(nr_dim, 4)
    Xs = [mwe.ops.normal_init(mwe.ops.allocate((nr_row, nr_dim)), nr_dim)
          for _ in range(100)]
    start = timer()
    ys, bp_ys = mwe.begin_update(Xs)
    dx = bp_ys(Xs)
    end = timer()
    print('Fast?', end, start, '%.4f' % (end-start))

    base = baseline_mwe(nr_dim, nr_piece, 4)
    start = timer()
    ys, bp_ys = base.begin_update(Xs)
    dx = bp_ys(Xs)
    end = timer()
    print('Slow?', end, start, '%.4f' % (end-start))


