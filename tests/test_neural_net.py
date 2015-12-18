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
    return [(ff, 0), (tf, 1), (ft, 1), (tt, 0)]


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
    small_eta_model = NeuralNet(2, 2, (10,10,10), rho=0.0, eta=0.0000001)
    normal_eta_model = NeuralNet(2, 2, (10,10,10), rho=0.0, eta=0.01)
    small_eta_loss = 0.0
    normal_eta_loss = 0.0
    for _ in range(100):
        for i, (features, label) in enumerate(xor_data):
            eg = small_eta_model.Example(features, gold=label)
            small_eta_loss += small_eta_model.train(eg)
            eg = normal_eta_model.Example(features, gold=label)
            normal_eta_loss += normal_eta_model.train(eg)
    assert normal_eta_loss < small_eta_loss


def test_xor_rho(xor_data):
    '''Test that higher L2 penalty causes slower learning.'''
    big_rho_model = NeuralNet(2, 2, (10,10,10), rho=0.8, eta=0.005)
    normal_rho_model = NeuralNet(2, 2, (10,10,10), rho=1e-4, eta=0.005)
    big_rho_loss = 0.0
    normal_rho_loss = 0.0
    for _ in range(100):
        for i, (features, label) in enumerate(xor_data):
            eg = big_rho_model.Example(features, gold=label)
            big_rho_loss += big_rho_model.train(eg)
            eg = normal_rho_model.Example(features, gold=label)
            normal_rho_loss += normal_rho_model.train(eg)
    assert normal_rho_loss < big_rho_loss


def test_xor_deep(xor_data):
    '''Compare 0, 1 and 3 layer networks.
    The 3 layer seems to do better, but it doesn't *have* to. But if the
    0 layer works, something's wrong!'''
    linear = NeuralNet(2, 2, tuple(), rho=0.0001, eta=0.005)
    small = NeuralNet(2, 2, (2,), rho=0.0001, eta=0.005)
    big = NeuralNet(2, 2, (10,10,10,10), rho=0.0001, eta=0.005)
    for _ in range(10000):
        for i, (features, label) in enumerate(xor_data):
            linear.train(linear.Example(features, gold=label))
            small.train(small.Example(features, gold=label))
            big.train(big.Example(features, gold=label))
        random.shuffle(xor_data)

    linear_loss = 0.0
    small_loss = 0.0
    big_loss = 0.0
    for i, (features, label) in enumerate(xor_data):
        eg = linear.Example(features, gold=label)
        linear(eg)
        linear_loss += 1 - eg.scores[label]
 
        eg = small.Example(features, gold=label)
        small(eg)
        small_loss += 1 - eg.scores[label]
            
        eg = big.Example(features, gold=label)
        big(eg)
        big_loss += 1 - eg.scores[label]
    # The deep network learns, the shallow small one doesn't, the linear one
    # can't
    assert big_loss < 0.5
    assert small_loss > 1.5
    assert linear_loss > 1.9
 

def test_model_widths(xor_data):
    '''Test different model widths'''
    w4 = NeuralNet(2, 2, (4,4), rho=0.0, eta=0.005)
    w4_w6 = NeuralNet(2, 2, (4, 6), rho=0.0, eta=0.005)
    assert w4_w6.nr_dense == (2*6+6) + (4*6+4) + (4*2+2)
    w4_loss = 0.0
    w4_w6_loss = 0.0
    for _ in range(100):
        for i, (features, label) in enumerate(xor_data):
            eg = w4.Example(features, gold=label)
            w4_loss += w4.train(eg)
            eg = w4_w6.Example(features, gold=label)
            w4_w6_loss += w4_w6.train(eg)
        random.shuffle(xor_data)
    # We don't know that the extra width is better, but it shouldn't be
    # *much* worse
    assert w4_w6_loss < (w4_loss * 1.1)
    # It also shouldn't be the same!
    assert w4_w6_loss != w4_loss
