from ...linear.sparse import SparseArray


def test_init():
    sp = SparseArray(10, 40.0)
    assert sp[10] == 40.0
    assert sp[0] == 0.0
    assert sp[11] == 0.0


def test_setitem():
    sp = SparseArray(10, 40.0)
    assert sp[10] == 40.0
    assert sp[0] == 0.0
    assert sp[11] == 0.0
    sp[0] = 12.0
    assert sp[0] == 12.0
    assert sp[10] == 40.0
    sp[0] = 14.0
    assert sp[0] == 14.0
    assert sp[10] == 40.0
    sp[52] = 6.0
    assert sp[0] == 14.0
    assert sp[1] == 0.0
    assert sp[10] == 40.0
    assert sp[52] == 6.0

def test_clone():
    sp1 = SparseArray(10, 40.0)
    sp2 = SparseArray(200, 2.)
    sp1 << sp2
    assert sp1[200] == 2.0
    assert sp2[200] == 2.
    sp1[2] = 2.
    assert sp1[2] == 2.
    assert 2 not in sp2
 
