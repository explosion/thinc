from __future__ import division

import pytest
import pickle
import io
import tempfile

from ...linear.avgtron import AveragedPerceptron
from ...extra.eg import Example


def assert_near_eq(float1, float2):
    assert abs(float1 - float2) < 0.001


def test_basic():
    nr_class = 3
    model = AveragedPerceptron(((1,), (2,), (3,), (4,), (5,)))
    instances = [
        (1, {1: 1, 3: -5}),
        (2, {2: 4, 3: 5})
    ]
    for clas, feats in instances:
        eg = Example(nr_class)
        #eg.features = feats
        #model(eg)
        #eg.costs = [i != clas for i in range(nr_class)]
        #model.update(eg)
    #eg = Example(nr_class)
    #eg.features = {1: 2, 2: 1}
    #model(eg)
    #assert eg.guess == 2
    #eg = Example(nr_class)
    #eg.features = {0: 2, 2: 1}
    #model(eg)
    #assert eg.scores[1] == 0
    #eg = Example(nr_class)
    #eg.features = {1: 2, 2: 1}
    #model(eg)
    #assert eg.scores[2] > 0
    #eg = Example(nr_class)
    #eg.features = {1: 2, 1: 1}
    #model(eg)
    #assert eg.scores[1] > 0
    #eg = Example(nr_class)
    #eg.features = {0: 3, 3: 1}
    #model(eg)
    #assert eg.scores[1] < 0 
    #eg = Example(nr_class)
    #eg.features = {0: 3, 3: 1}
    #model(eg)
    #assert eg.scores[2] > 0 

#
#@pytest.fixture
#def instances():
#    instances = [
#        [
#            (1, {1: -1, 2: 1}),
#            (2, {1: 5, 2: -5}),
#            (3, {1: 3, 2: -3}),
#        ],
#        [
#            (1, {1: -1, 2: 1}),
#            (2, {1: -1, 2: 2}),
#            (3, {1: 3, 2: -3})
#        ],
#        [
#            (1, {1: -1, 2: 2}),
#            (2, {1: 5, 2: -5}), 
#            (3, {4: 1, 5: -7, 2: 1})
#        ]
#    ]
#    return instances
#
#@pytest.fixture
#def model(instances):
#    templates = []
#    for batch in instances:
#        for _, feats in batch:
#            for key in feats:
#                templates.append((key,))
#    templates = tuple(set(templates))
#    model = AveragedPerceptron(templates)
#    for batch in instances:
#        model.time += 1
#        for clas, feats in batch:
#            for key, value in feats.items():
#                model.update_weight(key, clas, value)
#    return model
#
#def get_score(nr_class, model, feats, clas):
#    eg = Example(nr_class)
#    eg.features = feats
#    eg.costs = [i != clas for i in range(nr_class)]
#    model(eg)
#    return eg.scores[clas]
#
#
#def get_scores(nr_class, model, feats):
#    eg = Example(nr_class)
#    eg.features = feats
#    model(eg)
#    return list(eg.scores)
#
#
#def test_averaging(model):
#    model.end_training()
#    nr_class = 4
#    # Feature 1
#    assert_near_eq(get_score(nr_class, model, {1: 1}, 1), sum([-1, -2, -3]) / 3.0)
#    assert_near_eq(get_score(nr_class, model, {1: 1}, 2), sum([5, 4, 9]) / 3.0)
#    assert_near_eq(get_score(nr_class, model, {1: 1}, 3), sum([3, 6, 6]) / 3.0)
#    # Feature 2
#    assert_near_eq(get_score(nr_class, model, {2: 1}, 1), sum([1, 2, 4]) / 3.0)
#    assert_near_eq(get_score(nr_class, model, {2: 1}, 2), sum([-5, -3, -8]) / 3.0)
#    assert_near_eq(get_score(nr_class, model, {2: 1}, 3), sum([-3, -6, -5]) / 3.0)
#    # Feature 3 (absent)
#    assert_near_eq(get_score(nr_class, model, {3: 1}, 1), 0)
#    assert_near_eq(get_score(nr_class, model, {3: 1}, 2), 0)
#    assert_near_eq(get_score(nr_class, model, {3: 1}, 3), 0)
#    # Feature 4
#    assert_near_eq(get_score(nr_class, model, {4: 1}, 1), sum([0, 0, 0]) / 3.0)
#    assert_near_eq(get_score(nr_class, model, {4: 1}, 2), sum([0, 0, 0]) / 3.0)
#    assert_near_eq(get_score(nr_class, model, {4: 1}, 3), sum([0, 0, 1]) / 3.0)
#    # Feature 5
#    assert_near_eq(get_score(nr_class, model, {5: 1}, 1), sum([0, 0, 0]) / 3.0)
#    assert_near_eq(get_score(nr_class, model, {5: 1}, 2), sum([0, 0, 0]) / 3.0)
#    assert_near_eq(get_score(nr_class, model, {5: 1}, 3), sum([0, 0, -7]) / 3.0)
#
#
#def test_dump_load(model):
#    loc = tempfile.mkstemp()[1]
#    model.end_training()
#    model.dump(loc)
#    string = open(loc, 'rb').read()
#    assert string
#    new_model = AveragedPerceptron([(1,), (2,), (3,), (4,)])
#    nr_class = 5
#    assert get_scores(nr_class, model, {1: 1, 3: 1, 4: 1}) != \
#           get_scores(nr_class, new_model, {1:1, 3:1, 4:1})
#    assert get_scores(nr_class, model, {2:1, 5:1}) != \
#            get_scores(nr_class, new_model, {2:1, 5:1})
#    assert get_scores(nr_class, model, {2:1, 3:1, 4:1}) != \
#           get_scores(nr_class, new_model, {2:1, 3:1, 4:1})
#    new_model.load(loc)
#    assert get_scores(nr_class, model, {1:1, 3:1, 4:1}) == \
#           get_scores(nr_class, new_model, {1:1, 3:1, 4:1})
#    assert get_scores(nr_class, model, {2:1, 5:1}) == \
#           get_scores(nr_class, new_model, {2:1, 5:1})
#    assert get_scores(nr_class, model, {2:1, 3:1, 4:1}) == \
#           get_scores(nr_class, new_model, {2:1, 3:1, 4:1})
#
#
### TODO: Need a test that exercises multiple lines. Example bug:
### in gather_weights, don't increment f_i per row, only per feature
### (so overwrite some lines we're gathering)
