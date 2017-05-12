import pytest
from ..neural.util import get_ops
from ..neural.ops import NumpyOps, CupyOps


def test_get_ops():
    Ops = get_ops('numpy')
    Ops is NumpyOps
    Ops = get_ops('cpu')
    assert Ops is NumpyOps
    Ops = get_ops('cupy')
    assert Ops is CupyOps
    Ops = get_ops('gpu')
    assert Ops is CupyOps
    with pytest.raises(ValueError):
        Ops = get_ops('blah')


@pytest.mark.xfail
@pytest.mark.parametrize(
    'ids_batch', [
        (
            (
                ('the', 'cat', 'sat'),
                ('on', 'the', 'cat')
            ),
        ),
        (
            (
                ('the',),
                ('the',)
            ),
        ),
        (
            (
                ('a', 'b', 'a', 'b', 'd', 'e', 'f'),
            ),
        )
    ]
)
def test_get_positions(ids_batch):
    ids_batch = list(toolz.concat(ids_batch))
    positions = _get_positions(ids_batch)
    for key, idxs in positions.items():
        for i in idxs:
            assert ids_batch[i] == key


