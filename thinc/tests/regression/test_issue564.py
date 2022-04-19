from thinc.api import CupyOps
import torch


def test_issue564():
    if CupyOps.xp is not None:
        ops = CupyOps()
        t = torch.zeros((10, 2)).cuda()
        a = ops.asarray(t)

        assert a.shape == t.shape
        ops.xp.testing.assert_allclose(
            a,
            ops.alloc2f(10, 2),
        )
