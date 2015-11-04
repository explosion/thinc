from __future__ import division

import pytest

from thinc.learner import LinearModel


def assert_near_eq(float1, float2):
    assert abs(float1 - float2) < 0.001


def test_basic():
    model = LinearModel()
    model.update({1: {1: 1, 3: -5}, 2: {2: 4, 3: 5}}.items())
    assert model([(0, 1), (0, 1), (2, 1)])[0] == 0
    assert model([(0, 1), (0, 1), (2, 1)])[1] == 0
    assert model([(0, 1), (0, 1), (2, 1)])[2] > 0
    assert model([(0, 1), (1, 1), (0, 1)])[1] > 0
    assert model([(0, 1), (0, 1), (0, 1), (3, 1)])[1] < 0 
    assert model([(0, 1), (0, 1), (0, 1), (3, 1)])[2] > 0 

@pytest.fixture
def instances():
    instances = [
        {
            1: {1: -1, 2: 1},
            2: {1: 5, 2: -5},
            3: {1: 3, 2: -3},
        },
        {
            1: {1: -1, 2: 1},
            2: {1: -1, 2: 2},
            3: {1: 3, 2: -3},
        },
        {
            1: {1: -1, 2: 2},
            2: {1: 5, 2: -5}, 
            3: {4: 1, 5: -7, 2: 1}
        }
    ]
    return instances

@pytest.fixture
def model(instances):
    m = LinearModel()
    classes = range(3)
    for counts in instances:
        m.update(counts.items())
    return m

def test_averaging(model):
    model.end_training()
    # Feature 1
    assert_near_eq(model([(1, 1)])[1], sum([-1, -2, -3]) / 3.0)
    assert_near_eq(model([(1, 1)])[2], sum([5, 4, 9]) / 3.0)
    assert_near_eq(model([(1, 1)])[3], sum([3, 6, 6]) / 3.0)
    # Feature 2
    assert_near_eq(model([(2, 1)])[1], sum([1, 2, 4]) / 3.0)
    assert_near_eq(model([(2, 1)])[2], sum([-5, -3, -8]) / 3.0)
    assert_near_eq(model([(2, 1)])[3], sum([-3, -6, -5]) / 3.0)
    # Feature 3 (absent)
    assert_near_eq(model([(3, 1)])[1], 0)
    assert_near_eq(model([(3, 1)])[2], 0)
    assert_near_eq(model([(3, 1)])[3], 0)
    # Feature 4
    assert_near_eq(model([(4, 1)])[1], sum([0, 0, 0]) / 3.0)
    assert_near_eq(model([(4, 1)])[2], sum([0, 0, 0]) / 3.0)
    assert_near_eq(model([(4, 1)])[3], sum([0, 0, 1]) / 3.0)
    # Feature 5
    assert_near_eq(model([(5, 1)])[1], sum([0, 0, 0]) / 3.0)
    assert_near_eq(model([(5, 1)])[2], sum([0, 0, 0]) / 3.0)
    assert_near_eq(model([(5, 1)])[3], sum([0, 0, -7]) / 3.0)


def test_dump_load(model):
    loc = '/tmp/test_model'
    model.end_training()
    model.dump(loc)
    string = open(loc, 'rb').read()
    assert string
    new_model = LinearModel()
    assert list(model([(1, 1), (3, 1), (4, 1)])) != list(new_model([(1,1), (3, 1), (4, 1)]))
    assert list(model([(2, 1), (5, 1)])) != list(new_model([(2,1), (5, 1)]))
    assert list(model([(2, 1), (3,1), (4, 1)])) != list(new_model([(2,1), (3,1), (4,1)]))
    new_model.load(loc)
    assert list(model([(1, 1), (3, 1), (4, 1)])) == list(new_model([(1,1), (3, 1), (4,1)]))
    assert list(model([(2, 1), (5, 1)])) == list(new_model([(2,1), (5,1)]))
    assert list(model([(2, 1), (3, 1), (4, 1)])) == list(new_model([(2,1), (3,1), (4,1)]))


## TODO: Need a test that exercises multiple lines. Example bug:
## in gather_weights, don't increment f_i per row, only per feature
## (so overwrite some lines we're gathering)
