import pytest

from thinc.search.beam import Beam


def test_fill():
    b = Beam(5, 4)
    b.fill_from_list(
        [
            [1, 2, 3, 4, 5],
            [2, 4, 6, 8, 10],
            [4, 8, 12, 16, 20],
            [8, 16, 24, 32, 40]
        ]
    )

    i, clas = b.pop()
    assert i == 3
    assert clas == 4

    i, clas = b.pop()
    assert i == 3
    assert clas == 3

    assert len(b.extensions) == 2
    

def test_single_class():
    b = Beam(1, 2)
    b.fill_from_list([[10.7], [5.0]])
    i, clas = b.pop()
    assert i == 0
    assert clas == 0
    i, clas = b.pop()
    assert i == 1
    assert clas == 0
