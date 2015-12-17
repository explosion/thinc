from __future__ import division

import pytest
import pickle
import io
import numpy as np
import random
from numpy.testing import assert_allclose

from thinc.api import NeuralNet
from thinc.api import Example


def test_create():
    model = NeuralNet(3, 4, (8,))

    assert model.nr_class == 3
    assert model.nr_embed == 4
    assert model.nr_layer == 2
    assert model.layers == [(8, 4), (3, 8)]


def test_fwd_bias():
    model = NeuralNet(2, 2, tuple(), rho=0.0)
    assert model.nr_class == 2
    assert model.layers == [(2, 2)]

    model.set_weight(0, 0, 0, 1.0) # Weight of input 0, class 0
    model.set_weight(0, 0, 1, 1.0) # Weight of input 0, class 1
    model.set_weight(0, 1, 0, 1.0) # Weight of input 1, class 0
    model.set_weight(0, 1, 1, 1.0) # Weight of input 1, class 1


    model.set_bias(0, 0, 0.0) # Bias of class 0
    model.set_bias(0, 1, 0.0) # Bias of class 1
    eg = model.Example([])
    model(eg)
    assert_allclose(eg.scores, [0.5, 0.5])

    model.set_bias(0, 0, 100000.0)
    eg = model.Example([])
    model(eg)
    assert_allclose(eg.scores, [1.0, 0.0])

    model.set_bias(0, 0, 0.0)
    model.set_bias(0, 1, 100000.0)
    eg = model.Example([])
    model(eg)
    assert_allclose(eg.scores, [0.0, 1.0])

    model.set_bias(0, 0, 100000.0)
    model.set_bias(0, 1, 100000.0)
    eg = model.Example([])
    model(eg)
    assert_allclose(eg.scores, [0.5, 0.5])


def test_fwd_linear():
    model = NeuralNet(2, 2, tuple(), rho=0.0)
    assert model.nr_class == 2
    assert model.layers == [(2, 2)]

    model.set_weight(0, 0, 0, 1.0) # Weight of class 0, input 0
    model.set_weight(0, 0, 1, 0.0) # Weight of class 0, input 1
    model.set_weight(0, 1, 0, 0.0) # Weight of class 1, input 0
    model.set_weight(0, 1, 1, 1.0) # Weight of class 1, input 1


    model.set_bias(0, 0, 0.0) # Bias of class 0
    model.set_bias(0, 1, 0.0) # Bias of class 1
    # Awkward sparse representation
    ff = []
    ft = [(1, 1.0, 1, 1)]
    tf = [(1, 1.0, 0, 1)]
    tt = [(1, 1.0, 0, 1), (1, 1.0, 1, 1)]
    eg = model.Example([])
    model(eg)

    assert eg.activation(0, 0) == 0.0
    assert eg.activation(0, 1) == 0.0
    assert eg.activation(1, 0) == 0.5
    assert eg.activation(1, 1) == 0.5
    assert_allclose(eg.scores, [0.5, 0.5])

    eg = model.Example(ft)
    model(eg)
    
    assert eg.activation(0, 0) == 0.0
    assert eg.activation(0, 1) == 1.0
    assert_allclose([eg.activation(1, 0), eg.activation(1, 1)], [ 0.26894142,  0.73105858])
 
    assert_allclose([sum(eg.scores)], [1.0])

    eg = model.Example(tf)
    model(eg)
    assert_allclose([eg.activation(1, 0), eg.activation(1, 1)], [0.73105858, 0.26894142])
    assert_allclose(sum(eg.scores), [1.0])

    eg = model.Example(tt)
    model(eg)
    assert_allclose(eg.scores, [0.5, 0.5])


def test_xor():
    model = NeuralNet(2, 2, (2,), rho=0.0)
    assert model.nr_class == 2
    assert model.layers == [(2, 2), (2, 2)]

    # Make a network that detects X-or
    # It should output 0 if inputs are 0,0 or 1,1 and 1 if inputs are 0,1 or 1,0
    # A linear model can't do this!
    # What we do is learn two intermediate predictors, for 0,1 and 1,0
    # Then our output layer can detect either of these signals firing
    #
    # 0,0 --> neither fire
    # 0,1 --> A0 fires
    # 1,0 --> A1 fires
    # 1,1 --> neither fire
    #
    model.set_weight(0, 0, 0, 4.0) # Weight of A.0, in.0
    model.set_weight(0, 0, 1, -10.0)   # Weight of A.0, in.1
    model.set_weight(0, 1, 0, -10.0)   # Weight of A.1, in.0
    model.set_weight(0, 1, 1, 5.0) # Weight of A.1, in.1
    model.set_weight(1, 0, 0, -10.0)  # Weight of out.0, A.0
    model.set_weight(1, 0, 1, -10.0)  # Weight of out.0, A.1
    model.set_weight(1, 1, 0, 10.0)  # Weight of out.1, A.0
    model.set_weight(1, 1, 1, 10.0)  # Weight of out.1, A.1
    model.set_bias(0, 0, 0.0) # Bias of A 0
    model.set_bias(0, 1, 0.0) # Bias of A 1
    model.set_bias(1, 0, 10.0) # Bias of out 0
    model.set_bias(1, 1, -10.0) # Bias of out 1

    # Awkward sparse representation
    ff = []
    tf = [(1, 1.0, 0, 1)]
    ft = [(2, 1.0, 1, 1)]
    tt = [(1, 1.0, 0, 1), (2, 1.0, 1, 1)]

    eg = model.Example(ff)
    model(eg)
    assert eg.activation(0, 0) == 0.0
    assert eg.activation(0, 1) == 0.0
    assert eg.activation(1, 0) <= 0.0
    assert eg.activation(1, 1) <= 0.0
    assert eg.activation(2, 0) >  0.5
    assert eg.activation(2, 1) <  0.5
 
    eg = model.Example(tt)
    model(eg)
    assert eg.activation(0, 0) == 1.0
    assert eg.activation(0, 1) == 1.0
    assert eg.activation(1, 0) <= 0.0
    assert eg.activation(1, 1) <= 0.0
    assert eg.activation(2, 0) > 0.5
    assert eg.activation(2, 1) < 0.5
 
    eg = model.Example(tf)
    model(eg)
    assert eg.activation(0, 0) == 1.0
    assert eg.activation(0, 1) == 0.0
    assert eg.activation(1, 0) > 0.0
    assert eg.activation(1, 1) <= 0.0
    assert eg.activation(2, 0) < 0.5
    assert eg.activation(2, 1) > 0.5
 
    eg = model.Example(ft)
    model(eg)
    assert eg.activation(0, 0) == 0.0
    assert eg.activation(0, 1) == 1.0
    assert eg.activation(1, 0) <= 0.0
    assert eg.activation(1, 1) >= 0.0
    assert eg.activation(2, 0) < 0.5
    assert eg.activation(2, 1) > 0.5
 

#
#    #assert model.nr_class == 2
#    #assert model.nr_embed == 3
#    #assert model.nr_layer == 2
#    #assert model.layers == [(4, 3), (2, 4)]
#    #assert model.nr_dense == 26
#
#    #print(model.get_W(0))
#    #print(model.get_W(1))
#    #print(model.get_bias(0))
#    #print(model.get_bias(1))
#
#    #for _ in range(500):
#    #    for i, x in enumerate(Xs):
#    #        features = [(j, value, j, 1) for j, value in enumerate(x)]
#    #        prev = model.Example(features, gold=ys[i])
#    #        assert len(list(prev.scores)) == prev.nr_class == model.nr_class
#    #        assert sum(prev.scores) == 0
#    #        model(prev)
#    #        assert_allclose([sum(prev.scores)], [1.0])
#    #        eg = model.Example(features, gold=ys[i])
#    #        model.train(eg)
#    #        eg = model.Example(features, gold=ys[i])
#    #        model(eg)
#    #        print(i, ys[i], eg.best, list(prev.scores), list(eg.scores))
#    #        if prev.scores[ys[i]] != 1.0:
#    #            assert prev.scores[ys[i]] < eg.scores[ys[i]]
