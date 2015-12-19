from __future__ import division

import pytest
import pickle
import io
import numpy as np
import random
from numpy.testing import assert_allclose

from thinc.api import NeuralNet
from thinc.api import Example


#def test_create():
#    model = NeuralNet((4, 8, 3))
#
#    assert model.nr_in == 4
#    assert model.nr_out == 3
#    assert model.nr_layer == 3
#    assert model.widths == (4, 8, 3)
#
#
#def test_fwd_bias():
#    model = NeuralNet((2, 2), rho=0.0)
#    
#    assert model.nr_weight == 6
#    model.weights = [1.0] * model.nr_weight
#
#    scores = model([0, 0])
#    assert_allclose(scores, [0.5, 0.5])
#
#    # Set bias for class 0
#    model.weights = [1.,1.,1.,1.,100000.0,1.]
#    assert model.weights == [1.,1.,1.,1.,100000.0,1.]
#    scores = model([0, 0])
#    assert_allclose(scores, [1.0, 0.0])
#
#    # Set bias for class 1
#    model.weights = [1.,1.,1.,1.,1.,100000.0]
#    scores = model([0,0])
#    assert_allclose(scores, [0.0, 1.0])
#
#    # Set bias for both
#    model.weights = [1.,1.,1.,1.,100000.0,100000.0]
#    scores = model([0,0])
#    assert_allclose(scores, [0.5, 0.5])
#
#
#def test_fwd_linear():
#    model = NeuralNet((2,2), rho=0.0)
#    assert model.nr_out == 2
#    assert model.widths == (2, 2)
#
#    model.weights = [1.,0.,0.,1.,0.,0.]
#
#    ff = [0,0]
#    tf = [1,0]
#    ft = [0,1]
#    tt = [1,1]
#    scores = model(ff)
#
#    assert_allclose(scores, [0.5, 0.5])
#
#    scores = model(ft)
#    
#    assert_allclose(scores, [ 0.26894142,  0.73105858])
#    assert_allclose([sum(scores)], [1.0])
#
#    scores = model(tf)
#    assert_allclose(scores, [0.73105858, 0.26894142])
#    assert_allclose(sum(scores), [1.0])
#
#    scores = model(tt)
#    assert_allclose(scores, [0.5, 0.5])
#
#
#def test_xor_manual():
#    model = NeuralNet((2,2,2), rho=0.0)
#    assert model.nr_out == 2
#    assert model.widths == (2, 2, 2)
#
#    # Make a network that detects X-or
#    # It should output 0 if inputs are 0,0 or 1,1 and 1 if inputs are 0,1 or 1,0
#    # A linear model can't do this!
#    # 
#    # What we do is create two intermediate predictors, for 0,1 and 1,0
#    # These predictors rely on a bias towards 0. The non-linearity is essential
#    # Then our output layer can detect either of these signals firing
#    #
#    # 0,0 --> neither fire
#    # 0,1 --> A0 fires
#    # 1,0 --> A1 fires
#    # 1,1 --> neither fire
#    #
#    #model.set_weight(0, 0, 0, 4.0)    # Weight of A.0, in.0
#    #model.set_weight(0, 0, 1, -10.0)  # Weight of A.0, in.1
#    #model.set_weight(0, 1, 0, -10.0)  # Weight of A.1, in.0
#    #model.set_weight(0, 1, 1, 5.0)    # Weight of A.1, in.1
#    #model.set_weight(1, 0, 0, -10.0)  # Weight of out.0, A.0
#    #model.set_weight(1, 0, 1, -10.0)  # Weight of out.0, A.1
#    #model.set_weight(1, 1, 0, 10.0)   # Weight of out.1, A.0
#    #model.set_weight(1, 1, 1, 10.0)   # Weight of out.1, A.1
#    #model.set_bias(0, 0, 0.0)         # Bias of A 0
#    #model.set_bias(0, 1, 0.0)         # Bias of A 1
#    #model.set_bias(1, 0, 10.0)        # Bias of out 0
#    #model.set_bias(1, 1, -10.0)       # Bias of out 1
#
#
#    model.weights = np.asarray([
#                [4.0, -10.0],   # A.0*in.0, A.0*in.1
#                [-10.0, 5.0], # A.1*in.0, A.1*in.1
#                [0.0, 0.0],     # A.0 bias, A.1 bias
#                [-10.0, -10.0],  # out.0*A.0, out.0*A.1
#                [10.0, 10.0],   # out.1*A.0, out.1*A.1
#                [10.0, -10.0]   # out.0 bias, out.1 bias
#            ]).flatten()
#
#    ff = [0,0]
#    tf = [1,0]
#    ft = [0,1]
#    tt = [1,1]
#
#    scores = model(ff)
#    assert scores[0] > 0.99
# 
#    scores = model(tt)
#    assert scores[0] > 0.99
#    
#    scores = model(tf)
#    assert scores[1] > 0.99
#
#    scores = model(ft)
#    assert scores[1] > 0.99
# 

@pytest.fixture
def xor_data():
    ff = np.asarray([0.,0.], dtype='f')
    tf = np.asarray([1.,0.], dtype='f')
    ft = np.asarray([0.,1.], dtype='f')
    tt = np.asarray([1.,1.], dtype='f')
    costs0 = np.asarray([0., 1.], dtype='f')
    costs1 = np.asarray([1., 0.], dtype='f')
    return [(ff, 0, costs0), (tf, 1, costs1), (ft, 1, costs1), (tt, 0, costs0)]


@pytest.fixture
def or_data():
    ff = np.asarray([0.,0.], dtype='f')
    tf = np.asarray([1.,0.], dtype='f')
    ft = np.asarray([0.,1.], dtype='f')
    tt = np.asarray([1.,1.], dtype='f')
    costs0 = np.asarray([0., 1.], dtype='f')
    costs1 = np.asarray([1., 0.], dtype='f')
    return [(ff, 0, costs0), (tf, 1, costs1), (ft, 1, costs1), (tt, 1, costs1)]


def test_xor_gradient(xor_data):
    '''Test that after each update, we move towards the correct label.'''
    model = NeuralNet((2, 2, 2), rho=0.0, eta=1.0)

    assert model.nr_in == 2
    assert model.nr_out == 2
    assert model.nr_layer == 3
    
    for _ in range(500):
        for i, (features, label, costs) in enumerate(xor_data):
            prev = model(features)
            assert_allclose([sum(prev)], [1.0])
            model.train([(features, costs)])
            scores = model(features)
            assert prev[label] < scores[label] or prev[label] == scores[label] == 1.0


def test_xor_eta(xor_data):
    '''Test that a higher learning rate causes loss to decrease faster.'''
    small_eta_model = NeuralNet((2, 10,10,10, 2), rho=0.0, eta=0.0000001)
    normal_eta_model = NeuralNet((2, 10,10,10, 2), rho=0.0, eta=0.01)
    small_eta_loss = 0.0
    normal_eta_loss = 0.0
    for _ in range(100):
        for i, (features, label, costs) in enumerate(xor_data):
            small_eta_loss += small_eta_model.train([(features, costs)])
            normal_eta_loss += normal_eta_model.train([(features, costs)])
    assert normal_eta_loss < small_eta_loss


def test_xor_rho(xor_data):
    '''Test that higher L2 penalty causes slower learning.'''
    big_rho_model = NeuralNet((2,10,10,10,2), rho=0.8, eta=0.005)
    normal_rho_model = NeuralNet((2, 10,10,10, 2), rho=1e-4, eta=0.005)
    big_rho_model.weights = list(normal_rho_model.weights)
    big_rho_loss = 0.0
    normal_rho_loss = 0.0
    for _ in range(10):
        for i, (features, label, costs) in enumerate(xor_data):
            big_rho_loss += big_rho_model.train([(features, costs)])
            normal_rho_loss += normal_rho_model.train([(features, costs)])
    assert normal_rho_loss < big_rho_loss


def test_xor_deep(xor_data):
    '''Compare 0, 1 and 3 layer networks.
    The 3 layer seems to do better, but it doesn't *have* to. But if the
    0 layer works, something's wrong!'''
    linear = NeuralNet((2,2), rho=0.0001, eta=0.005)
    small = NeuralNet((2,2,2), rho=0.0001, eta=0.005)
    big = NeuralNet((2,10,10,10,10,2), rho=0.0001, eta=0.005)
    for _ in range(10000):
        for i, (features, label, costs) in enumerate(xor_data):
            linear.train([(features, costs)])
            big.train([(features, costs)])
            small.train([(features, costs)])
        random.shuffle(xor_data)

    linear_loss = 0.0
    small_loss = 0.0
    big_loss = 0.0
    for i, (features, label, costs) in enumerate(xor_data):
        scores = linear(features)
        linear_loss += 1 - scores[label]
 
        scores = small(features)
        small_loss += 1 - scores[label]
            
        scores = big(features)
        big_loss += 1 - scores[label]
    # The deep network learns, the shallow small one doesn't, the linear one
    # can't
    assert big_loss < 0.5
    assert small_loss > 1.5
    assert linear_loss > 1.9
 

def test_model_widths(or_data):
    '''Test different model widths'''
    w4 = NeuralNet((2,4,2), rho=0.0, eta=0.005)
    w4_w6 = NeuralNet((2,20,2), rho=0.0, eta=0.005)
    assert w4_w6.nr_weight > w4.nr_weight
    w4_loss = 0.0
    w4_w6_loss = 0.0
    for _ in range(100):
        for i, (features, label, costs) in enumerate(or_data):
            w4_loss += w4.train([(features, costs)])
            w4_w6_loss += w4_w6.train([(features, costs)])
        random.shuffle(or_data)
    # We don't know that the extra width is better, but it shouldn't be
    # *much* worse
    assert w4_w6_loss < (w4_loss * 1.1)
    # It also shouldn't be the same!
    assert w4_w6_loss != w4_loss
