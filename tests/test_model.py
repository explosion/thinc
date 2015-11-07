from __future__ import division

import pytest
import pickle
import io

from thinc.features import Extracter
from thinc.api import AveragedPerceptron
from thinc.api import Example


def assert_near_eq(float1, float2):
    assert abs(float1 - float2) < 0.001


def test_basic():
    model = AveragedPerceptron(3, Extracter())

    instance = {1: {1: 1, 3: -5}, 2: {2: 4, 3: 5}}
    for clas, feats in instance.items():
        eg = Example.from_feats(model.nr_class, feats.items(), gold=clas)
        model(eg)
        model.updater(eg)
    eg = Example.from_feats(3, [(0, 1), (0, 1), (2, 1)])
    model(eg)
    assert eg.guess == 2
    eg = Example.from_feats(3, [(0, 1), (0, 1), (2, 1)])
    model(eg)
    assert eg.scores[1] == 0
    eg = Example.from_feats(3, [(0, 1), (0, 1), (2, 1)])
    model(eg)
    assert eg.scores[2] > 0
    eg = Example.from_feats(3, [(0, 1), (1, 1), (0, 1)])
    model(eg)
    assert eg.scores[1] > 0
    eg = Example.from_feats(3, [(0, 1), (0, 1), (0, 1), (3, 1)])
    model(eg)
    assert eg.scores[1] < 0 
    eg = Example.from_feats(3, [(0, 1), (0, 1), (0, 1), (3, 1)])
    model(eg)
    assert eg.scores[2] > 0 


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
    model = AveragedPerceptron(4, Extracter())
    for counts in instances:
        model.updater.time += 1
        for clas, feats in counts.items():
            for key, value in feats.items():
                model.updater.update_weight(key, clas, value)
    return model

def get_score(model, feats, clas):
    eg = Example.from_feats(model.nr_class, feats, gold=clas)
    model(eg)
    return eg.scores[clas]


def get_scores(model, feats):
    eg = Example.from_feats(model.nr_class, feats)
    model(eg)
    return list(eg.scores)


def test_averaging(model):
    model.updater.end_training()
    # Feature 1
    assert_near_eq(get_score(model, [(1, 1)], 1), sum([-1, -2, -3]) / 3.0)
    assert_near_eq(get_score(model, [(1, 1)], 2), sum([5, 4, 9]) / 3.0)
    assert_near_eq(get_score(model, [(1, 1)], 3), sum([3, 6, 6]) / 3.0)
    # Feature 2
    assert_near_eq(get_score(model, [(2, 1)], 1), sum([1, 2, 4]) / 3.0)
    assert_near_eq(get_score(model, [(2, 1)], 2), sum([-5, -3, -8]) / 3.0)
    assert_near_eq(get_score(model, [(2, 1)], 3), sum([-3, -6, -5]) / 3.0)
    # Feature 3 (absent)
    assert_near_eq(get_score(model, [(3, 1)], 1), 0)
    assert_near_eq(get_score(model, [(3, 1)], 2), 0)
    assert_near_eq(get_score(model, [(3, 1)], 3), 0)
    # Feature 4
    assert_near_eq(get_score(model, [(4, 1)], 1), sum([0, 0, 0]) / 3.0)
    assert_near_eq(get_score(model, [(4, 1)], 2), sum([0, 0, 0]) / 3.0)
    assert_near_eq(get_score(model, [(4, 1)], 3), sum([0, 0, 1]) / 3.0)
    # Feature 5
    assert_near_eq(get_score(model, [(5, 1)], 1), sum([0, 0, 0]) / 3.0)
    assert_near_eq(get_score(model, [(5, 1)], 2), sum([0, 0, 0]) / 3.0)
    assert_near_eq(get_score(model, [(5, 1)], 3), sum([0, 0, -7]) / 3.0)


def test_dump_load(model):
    loc = '/tmp/test_model'
    model.updater.end_training()
    model.dump(loc)
    string = open(loc, 'rb').read()
    assert string
    new_model = AveragedPerceptron(model.nr_class, Extracter())
    assert get_scores(model, [(1, 1), (3, 1), (4, 1)]) != get_scores(new_model, [(1,1), (3, 1), (4, 1)])
    assert get_scores(model, [(2, 1), (5, 1)]) != get_scores(new_model, [(2,1), (5, 1)])
    assert get_scores(model, [(2, 1), (3,1), (4, 1)]) != get_scores(new_model, [(2,1), (3,1), (4,1)])
    new_model.load(loc)
    assert get_scores(model, [(1, 1), (3, 1), (4, 1)]) == get_scores(new_model, [(1,1), (3, 1), (4,1)])
    assert get_scores(model, [(2, 1), (5, 1)]) == get_scores(new_model, [(2,1), (5,1)])
    assert get_scores(model, [(2, 1), (3, 1), (4, 1)]) == get_scores(new_model, [(2,1), (3,1), (4,1)])


def test_pickle(model):
    file_ = io.BytesIO()
    pickle.dump(model, file_)
    file_.seek(0)
    loaded = pickle.load(file_)




## TODO: Need a test that exercises multiple lines. Example bug:
## in gather_weights, don't increment f_i per row, only per feature
## (so overwrite some lines we're gathering)
