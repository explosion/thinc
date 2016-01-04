from __future__ import division, print_function

import pytest
import pickle
import io
import numpy as np
import random
from numpy.testing import assert_allclose

from thinc.nn import NeuralNet
from thinc.eg import Example, Batch

np.random.seed(2)


def test_create():
    model = NeuralNet((4, 8, 3))

    assert model.nr_in == 4
    assert model.nr_out == 3
    assert model.nr_layer == 3
    assert model.widths == (4, 8, 3)


def test_fwd_bias():
    model = NeuralNet((2, 2), rho=0.0)
    
    assert model.nr_weight == 10
    model.weights = [1.0] * model.nr_weight

    eg = model([0, 0])
    assert_allclose(eg.scores, [0.5, 0.5])

    # Set bias for class 0
    syn = [1.,1.,1.,1.]
    bias = [100000.,1.]
    gamma = [0,0]
    beta = [0,0]
    model.weights = syn + bias + gamma + beta
    assert model.weights == list(syn + bias + gamma + beta)
    eg = model([0, 0])
    assert_allclose(eg.scores, [1.0, 0.0])

    # Set bias for class 1
    model.weights = syn + [1.,10000.0] + gamma + beta
    eg = model([0,0])
    assert_allclose(eg.scores, [0.0, 1.0])

    # Set bias for both
    model.weights = syn + [10000.0,10000.0] + gamma + beta
    eg = model([0,0])
    assert_allclose(eg.scores, [0.5, 0.5])


def test_fwd_linear():
    model = NeuralNet((2,2), rho=0.0)
    assert model.nr_out == 2
    assert model.widths == (2, 2)

    syn = [1.,0.,0.,1.]
    bias = [0.,0.]
    gamma = [0.,0.]
    beta = [0.,0.]
    model.weights = syn+bias+gamma+beta

    ff = [0,0]
    tf = [1,0]
    ft = [0,1]
    tt = [1,1]
    eg = model(ff)

    assert_allclose(eg.scores, [0.5, 0.5])

    eg = model(ft)
    
    assert_allclose(eg.scores, [ 0.26894142,  0.73105858])
    assert_allclose([sum(eg.scores)], [1.0])

    eg = model(tf)
    assert_allclose(eg.scores, [0.73105858, 0.26894142])
    assert_allclose(sum(eg.scores), [1.0])

    eg = model(tt)
    assert_allclose(eg.scores, [0.5, 0.5])


def test_xor_manual():
    model = NeuralNet((2,2,2), rho=0.0)
    assert model.nr_out == 2
    assert model.widths == (2, 2, 2)

    # Make a network that detects X-or
    # It should output 0 if inputs are 0,0 or 1,1 and 1 if inputs are 0,1 or 1,0
    # A linear model can't do this!
    # 
    # What we do is create two intermediate predictors, for 0,1 and 1,0
    # These predictors rely on a bias towards 0. The non-linearity is essential
    # Then our output layer can detect either of these signals firing
    #
    # 0,0 --> neither fire
    # 0,1 --> A0 fires
    # 1,0 --> A1 fires
    # 1,1 --> neither fire
    #

    model.weights = np.asarray([
                [4.0, -10.0],   # A.0*in.0, A.0*in.1
                [-10.0, 5.0], # A.1*in.0, A.1*in.1
                [0.0, 0.0],     # A.0 bias, A.1 bias
                [1.0, 1.0],     # A.0 gamma, A.1 gamma
                [0.0, 0.0],     # A.0 beta, A.1 beta
                [-10.0, -10.0],  # out.0*A.0, out.0*A.1
                [10.0, 10.0],   # out.1*A.0, out.1*A.1
                [10.0, -10.0],   # out.0 bias, out.1 bias
                [1.0, 1.0],     # out.0 gamma, out.1 gamma
                [0.0, 0.0],     # out.0 beta, out.1 beta
            ]).flatten()

    ff = [0,0]
    tf = [1,0]
    ft = [0,1]
    tt = [1,1]

    eg = model(ff)
    assert eg.scores[0] > 0.99
 
    eg = model(tt)
    assert eg.scores[0] > 0.99
    
    eg = model(tf)
    assert eg.scores[1] > 0.99

    eg = model(ft)
    assert eg.scores[1] > 0.99
 

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


def test_learn_linear(or_data):
    '''Test that a linear model can learn OR.'''
    # Need high eta on this sort of toy problem, or learning takes forever!
    model = NeuralNet((2, 2), rho=0.0, eta=0.1, eps=1e-4)

    assert model.nr_in == 2
    assert model.nr_out == 2
    assert model.nr_layer == 2
    
    # It takes about this many iterations, with the settings above.
    for _ in range(50):
        for feats, label, costs in or_data:
            batch = model.train([feats], [costs])
        random.shuffle(or_data)
    acc = 0.0
    for features, label, costs in or_data:
        eg = model(features)
        assert costs[label] == 0
        acc += eg.scores[label] > 0.5
    assert acc == len(or_data)


def test_mlp_learn_linear(or_data):
    '''Test that with a hidden layer, we can still learn OR'''
    # Need high eta on this sort of toy problem, or learning takes forever!
    model = NeuralNet((2, 3, 2), rho=0.0, eta=0.5, eps=1e-4, bias=0.0)

    assert model.nr_in == 2
    assert model.nr_out == 2
    assert model.nr_layer == 3
    
    # Keep this set low, so that we see that the hidden layer allows the function
    # to be learned faster than the linear model
    for _ in range(50):
        for feats, label, costs in or_data:
            batch = model.train([feats], [costs])
        random.shuffle(or_data)
    acc = 0.0
    for features, label, costs in or_data:
        eg = model(features)
        assert costs[label] == 0
        acc += eg.scores[label] > 0.5
    assert acc == len(or_data)


def test_xor_gradient(xor_data):
    '''Test that after each update, we move towards the correct label.'''
    model = NeuralNet((2, 2, 2), rho=0.0, eta=1.0)

    assert model.nr_in == 2
    assert model.nr_out == 2
    assert model.nr_layer == 3
    
    for _ in range(500):
        for i, (features, label, costs) in enumerate(xor_data):
            prev = model(features)
            assert_allclose([sum(prev.scores)], [1.0])
            model.train([features], [costs]).loss
            eg = model(features)
            assert (prev.scores[label] < eg.scores[label] or \
                    prev.scores[label] == eg.scores[label] == 1.0)


def test_xor_eta(xor_data):
    '''Test that a higher learning rate causes loss to decrease faster.'''
    small_eta_model = NeuralNet((2, 10,10,10, 2), rho=0.0, eta=0.0000001)
    normal_eta_model = NeuralNet((2, 10,10,10, 2), rho=0.0, eta=0.1)
    small_eta_loss = 0.0
    normal_eta_loss = 0.0
    for _ in range(100):
        for i, (features, label, costs) in enumerate(xor_data):
            small_eta_loss += small_eta_model.train([features], [costs]).loss
            normal_eta_loss += normal_eta_model.train([features], [costs]).loss
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
            big_rho_loss += big_rho_model.train([features], [costs]).loss
            normal_rho_loss += normal_rho_model.train([features], [costs]).loss
    assert normal_rho_loss < (big_rho_loss * 1.1)


def test_xor_deep(xor_data):
    '''Compare 0, 1 and 3 layer networks.
    The 3 layer seems to do better, but it doesn't *have* to. But if the
    0 layer works, something's wrong!'''
    linear = NeuralNet((2,2), rho=0.0001, eta=0.005)
    small = NeuralNet((2,2,2), rho=0.0001, eta=0.005)
    big = NeuralNet((2,10,10,10,10,2), rho=0.0001, eta=0.005)
    for _ in range(10000):
        for i, (features, label, costs) in enumerate(xor_data):
            linear.train([features], [costs]).loss
            big.train([features], [costs]).loss
            scores = big(features)
            small.train([features], [costs]).loss
        random.shuffle(xor_data)

    linear_loss = 0.0
    small_loss = 0.0
    big_loss = 0.0
    for i, (features, label, costs) in enumerate(xor_data):
        linear_loss += 1 - linear(features).scores[label]
        small_loss += 1 - small(features).scores[label]
        big_loss += 1 - big(features).scores[label]
    # The deep network learns, the shallow small one doesn't, the linear one
    # can't
    assert big_loss < 0.5
    assert small_loss > 1.0
    assert linear_loss > 1.9
 

def test_model_widths(or_data):
    '''Test different model widths'''
    narrow = NeuralNet((2,4,2), rho=0.0, eta=0.005)
    wide = NeuralNet((2,20,2), rho=0.0, eta=0.005)
    assert wide.nr_weight > narrow.nr_weight
    narrow_loss = 0.0
    wide_loss = 0.0
    for _ in range(100):
        for i, (features, label, costs) in enumerate(or_data):
            narrow_loss += narrow.train([features], [costs]).loss
            wide_loss += wide.train([features], [costs]).loss
        random.shuffle(or_data)
    # We don't know that the extra width is better, but it shouldn't be
    # *much* worse
    assert wide_loss < (narrow_loss * 1.1)
    # It also shouldn't be the same!
    #assert wide_loss != narrow_loss


def test_embedding():
    model = NeuralNet((10,4,2), embed=((5,), (0,0)), rho=0.0, eta=0.005)
    assert model.nr_in == 10
    eg = model.Example({(0, 1): 2.5})
    model(eg)
    assert eg.activation(0, 0) != 0
    assert eg.activation(0, 1) != 0
    assert eg.activation(0, 2) != 0
    assert eg.activation(0, 3) != 0
    assert eg.activation(0, 4) != 0
    
    eg = model.Example({(1, 1867): 0.5})
    model(eg)
    assert eg.activation(0, 0) == 0.0
    assert eg.activation(0, 1) == 0.0
    assert eg.activation(0, 2) == 0.0
    assert eg.activation(0, 3) == 0.0
    assert eg.activation(0, 4) == 0.0
    assert eg.activation(0, 5) != 0.0
    assert eg.activation(0, 6) != 0.0
    assert eg.activation(0, 7) != 0.0
    assert eg.activation(0, 8) != 0.0
    assert eg.activation(0, 9) != 0.0
    

def test_simple_backprop():
    model = NeuralNet((2, 3, 2))
    syn1 = [-2.6, 1.59, 0.09, 1.23, 0.63, 0.01]
    bias1 = [0.0 for _ in range(model.widths[1])]
    beta1 = [0.0 for _ in range(model.widths[1])]
    gamma1 = [0.0 for _ in range(model.widths[1])]
    syn2 = [0.0 for _ in range(model.widths[1] * model.widths[2])]
    bias2 = [0.0 for _ in range(model.widths[2])]
    beta2 = [0.0 for _ in range(model.widths[2])]
    gamma2 = [0.0 for _ in range(model.widths[2])]
    model.weights = syn1 + bias1 + beta1 + gamma1 + syn2 + bias2 + beta2 + gamma2
    for w, b in model.layers:
        print('W', w)
        print('b', b)
    batch = model.train([(0, 1)], [(1, 0)])
    print(batch.gradient)
