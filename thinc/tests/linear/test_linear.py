# coding: utf8
from __future__ import unicode_literals, division

import numpy
import pytest

from ...linear.linear import LinearModel
from ...neural.optimizers import SGD
from ...neural.ops import NumpyOps
from ...neural.util import to_categorical


@pytest.fixture
def instances():
    lengths = numpy.asarray([5, 4], dtype="int32")
    keys = numpy.arange(9, dtype="uint64")
    values = numpy.ones(9, dtype="float")
    X = (keys, values, lengths)
    y = numpy.asarray([0, 2], dtype="int32")
    return X, to_categorical(y, nb_classes=3)


@pytest.fixture
def sgd():
    return SGD(NumpyOps(), 0.001)


@pytest.mark.xfail
def test_basic(instances, sgd):
    X, y = instances
    nr_class = 3
    model = LinearModel(nr_class)
    yh, backprop = model.begin_update(X)
    loss1 = ((yh - y) ** 2).sum()
    backprop(yh - y, sgd)
    yh, backprop = model.begin_update(X)
    loss2 = ((yh - y) ** 2).sum()
    assert loss2 < loss1
    print(loss2, loss1)


# @pytest.fixture
# def model(instances):
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
# def get_score(nr_class, model, feats, clas):
#    eg = Example(nr_class)
#    eg.features = feats
#    eg.costs = [i != clas for i in range(nr_class)]
#    model(eg)
#    return eg.scores[clas]
#
#
# def get_scores(nr_class, model, feats):
#    eg = Example(nr_class)
#    eg.features = feats
#    model(eg)
#    return list(eg.scores)
#
#
# def test_averaging(model):
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
# def test_dump_load(model):
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
