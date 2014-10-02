from __future__ import division

import StringIO
import pytest

from thinc.ml.learner import LinearModel


def test_basic():
    model = LinearModel(7, 4)
    model.update({0: {(1, 1): 1, (3, 3): -5}, 1: {(2, 2): 4, (3, 3): 5}})
    assert model([0, 0, 2])[:2] == [0, 4]
    assert model([0, 1])[:2] == [1, 0]
    assert model([0, 0, 0, 3])[:2] == [-5, 5]
    scores = model([0, 1, 2, 3, 0, 0])
    assert scores[0] == sum([1, 0, -5])
    assert scores[1] == sum([0, 4, 5])


@pytest.fixture
def instances():
    instances = [
        {
            0: {(1, 1): -1, (2, 2): 1},
            1: {(1, 1): 5, (2, 2): -5},
            2: {(1, 1): 3, (2, 2): -3},
        },
        {
            0: {(1, 1): -1, (2, 2): 1},
            1: {(1, 1): -1, (2, 2): 2},
            2: {(1, 1): 3, (2, 2): -3},
        },
        {
            0: {(1, 1): -1, (2, 2): 2},
            1: {(1, 1): 5, (2, 2): -5}, 
            2: {(4, 4): 1, (5, 5): -7, (2, 2): 1}
        }
    ]
    return instances


@pytest.fixture
def model(instances):
    m = LinearModel(3, 6)
    classes = range(3)
    for counts in instances:
        m.update(counts)
    return m

def test_averaging(model):
    model.end_training()
    # Feature 1
    assert model([0, 1])[0] == sum([-1, -2, -3]) / 1
    assert model([0, 1])[1] == sum([5, 4, 9]) / 1
    assert model([0, 1])[2] == sum([3, 6, 6]) / 1
    # Feature 2
    assert model([0, 0, 2])[0] == sum([1, 2, 4]) / 1
    assert model([0, 0, 2])[1] == sum([-5, -3, -8]) / 1
    assert model([0, 0, 2])[2] == sum([-3, -6, -5]) / 1
    # Feature 3 (absent)
    assert model([0, 0, 0, 3])[0] == 0
    assert model([0, 0, 0, 3])[1] == 0
    assert model([0, 0, 0, 3])[2] == 0
    # Feature 4
    assert model([0, 0, 0, 0, 4])[0] == sum([0, 0, 0]) / 1
    assert model([0, 0, 0, 0, 4])[1] == sum([0, 0, 0]) / 1
    assert model([0, 0, 0, 0, 4])[2] == sum([0, 0, 1]) / 1
    # Feature 5
    assert model([0, 0, 0, 0, 0, 5])[0] == sum([0, 0, 0]) / 1
    assert model([0, 0, 0, 0, 0, 5])[1] == sum([0, 0, 0]) / 1
    assert model([0, 0, 0, 0, 0, 5])[2] == sum([0, 0, -7]) / 1


def test_dump_load(model):
    output = StringIO.StringIO()
    model.dump(output)
    string = output.getvalue()
    assert string
    new_model = LinearModel(3, 6)
    assert model([0, 1, 0, 3, 4]) != new_model([0, 1, 0, 3, 4])
    assert model([0, 0, 2, 0, 0, 5]) != new_model([0, 0, 2, 0, 0, 5])
    assert model([0, 0, 2, 3, 4]) != new_model([0, 0, 2, 3, 4])
    new_model.load(StringIO.StringIO(string))
    assert model([0, 1, 0, 3, 4]) == new_model([0, 1, 0, 3, 4])
    assert model([0, 0, 2, 0, 0, 5]) == new_model([0, 0, 2, 0, 0, 5])
    assert model([0, 0, 2, 3, 4]) == new_model([0, 0, 2, 3, 4])
