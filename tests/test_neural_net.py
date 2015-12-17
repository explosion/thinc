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
    tf = [(1, 1.0, 0, 1)]
    ft = [(2, 1.0, 1, 1)]
    tt = [(1, 1.0, 0, 1), (2, 1.0, 1, 1)]
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


def test_xor_manual():
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
 

@pytest.fixture
def xor_data():
    # Awkward sparse representation
    ff = []
    tf = [(1, 1.0, 0, 1)]
    ft = [(2, 1.0, 1, 1)]
    tt = [(1, 1.0, 0, 1), (2, 1.0, 1, 1)]
    return [(ff, 0), (tf, 1), (ft, 1), (tt, 1)]


def test_xor_gradient(xor_data):
    '''Test that after each update, we move towards the correct label.'''
    model = NeuralNet(2, 2, (2,), rho=0.0)
    assert model.nr_class == 2
    assert model.layers == [(2, 2), (2, 2)]

    assert model.nr_class == 2
    assert model.nr_embed == 2
    assert model.nr_layer == 2
    
    for _ in range(500):
        for i, (features, label) in enumerate(xor_data):
            prev = model.Example(features, gold=label)
            assert len(list(prev.scores)) == prev.nr_class == model.nr_class
            assert sum(prev.scores) == 0
            model(prev)
            assert_allclose([sum(prev.scores)], [1.0])
            eg = model.Example(features, gold=label)
            model.train(eg)
            eg = model.Example(features, gold=label)
            model(eg)
            if prev.scores[label] != 1.0:
                assert prev.scores[label] < eg.scores[label]

def test_xor_eta(xor_data):
    '''Test that a higher learning rate causes loss to decrease faster.'''
    eta_005_model = NeuralNet(2, 2, (2,), rho=0.0, eta=0.005)
    eta_01_model = NeuralNet(2, 2, (2,), rho=0.0, eta=0.01)
    eta_005_loss = 0.0
    eta_01_loss = 0.0
    for _ in range(5):
        for i, (features, label) in enumerate(xor_data):
            eg = eta_005_model.Example(features, gold=label)
            eta_005_loss += eta_005_model.train(eg)
            eg = eta_01_model.Example(features, gold=label)
            eta_01_loss += eta_01_model.train(eg)
    assert eta_01_loss < eta_005_loss


def test_xor_rho(xor_data):
    '''Test that higher L2 penalty causes slower learning.'''
    rho_0001_model = NeuralNet(2, 2, (2,), rho=0.0001, eta=0.005)
    rho_001_model = NeuralNet(2, 2, (2,), rho=0.001, eta=0.005)
    rho_0_model = NeuralNet(2, 2, (2,), rho=0.0, eta=0.005)
    rho_0001_loss = 0.0
    rho_001_loss = 0.0
    rho_0_loss = 0.0
    for _ in range(10):
        for i, (features, label) in enumerate(xor_data):
            eg = rho_0001_model.Example(features, gold=label)
            rho_0001_loss += rho_0001_model.train(eg)
            
            eg = rho_001_model.Example(features, gold=label)
            rho_001_loss += rho_001_model.train(eg)
            
            eg = rho_0_model.Example(features, gold=label)
            rho_0_loss += rho_0_model.train(eg)
    assert rho_0_loss < rho_0001_loss
    assert rho_0001_loss < rho_001_loss
 
