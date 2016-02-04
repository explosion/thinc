from __future__ import division, print_function

import pytest
import pickle
import io
import numpy as np
import random
from numpy.testing import assert_allclose

from thinc.neural.nn import NeuralNet
from thinc.extra.eg import Example

np.random.seed(2)
random.seed(0)


def test_create():
    model = NeuralNet((4, 8, 3))

    assert model.nr_in == 4
    assert model.nr_class == 3
    assert model.nr_layer == 3
    assert model.widths == (4, 8, 3)
    assert model.mem is not None

    with pytest.raises(AttributeError) as excinfo:
        model.mem = None

def test_fwd_bias():
    model = NeuralNet((2, 2), rho=0.0)
    
    model.weights = [1.0] * model.nr_weight

    eg = model.predict_dense([0, 0])
    assert eg.nr_class == 2
    assert_allclose(eg.scores, [0.5, 0.5])

    # Set bias for class 0
    syn = [1.,1.,1.,1.]
    bias = [100000.,1.]
    gamma = [0,0]
    if model.nr_weight == len(syn) + len(bias):
        model.weights = syn + bias
        assert model.weights == list(syn + bias)
    else:
        model.weights = syn + bias + gamma
        assert model.weights == list(syn + bias + gamma)
    eg = model.predict_dense([0, 0])
    assert_allclose(eg.scores, [1.0, 0.0])

    # Set bias for class 1
    if model.nr_weight == 6:
        model.weights = syn + [1.,10000.0]
    else:
        model.weights = syn + [1.,10000.0] + gamma
    eg = model.predict_dense([0,0])
    assert_allclose(eg.scores, [0.0, 1.0])

    # Set bias for both
    if model.nr_weight == 6:
        model.weights = syn + [10000.0,10000.0]
    else:
        model.weights = syn + [10000.0,10000.0] + gamma
    eg = model.predict_dense([0,0])
    assert_allclose(eg.scores, [0.5, 0.5])


def test_fwd_linear():
    model = NeuralNet((2,2), rho=0.0, alpha=0.5)
    assert model.nr_class == 2
    assert model.widths == (2, 2)

    syn = [1.,0.,0.,1.]
    bias = [0.,0.]
    gamma = [1.,1.]
    if model.use_batch_norm:
        model.weights = syn+bias+gamma
    else:
        model.weights = syn+bias
    
    ff = [0,0]
    tf = [1,0]
    ft = [0,1]
    tt = [1,1]
    eg = model.predict_dense(ff)

    assert_allclose(eg.scores, [0.5, 0.5])

    eg = model.predict_dense(ft)
    
    assert_allclose(eg.scores, [ 0.26894142,  0.73105858])
    assert_allclose([sum(eg.scores)], [1.0])

    eg = model.predict_dense(tf)
    assert_allclose(eg.scores, [0.73105858, 0.26894142])
    assert_allclose(sum(eg.scores), [1.0])

    eg = model.predict_dense(tt)

    assert_allclose(eg.scores, [0.5, 0.5])


def test_xor_manual():
    model = NeuralNet((2,2,2), rho=0.0)
    assert model.nr_class == 2
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

    if not model.use_batch_norm:
        model.weights = np.asarray([
                    [4.0, -10.0],   # A.0*in.0, A.0*in.1
                    [-10.0, 5.0], # A.1*in.0, A.1*in.1
                    [0.0, 0.0],     # A.0 bias, A.1 bias
                    [-10.0, -10.0],  # out.0*A.0, out.0*A.1
                    [10.0, 10.0],   # out.1*A.0, out.1*A.1
                    [10.0, -10.0],   # out.0 bias, out.1 bias
                ]).flatten()
    else:
        model.weights = np.asarray([
                    [4.0, -10.0],   # A.0*in.0, A.0*in.1
                    [-10.0, 5.0], # A.1*in.0, A.1*in.1
                    [0.0, 0.0],     # A.0 bias, A.1 bias
                    [1.0, 1.0],     # A.0 gamma, A.1 gamma
                    [-10.0, -10.0],  # out.0*A.0, out.0*A.1
                    [10.0, 10.0],   # out.1*A.0, out.1*A.1
                    [10.0, -10.0],   # out.0 bias, out.1 bias
                    [1.0, 1.0],     # out.0 gamma, out.1 gamma
                ]).flatten()

    ff = [0,0]
    tf = [1,0]
    ft = [0,1]
    tt = [1,1]

    eg = model.predict_dense(ff)
    assert eg.scores[0] > 0.99
 
    eg = model.predict_dense(tt)
    assert eg.scores[0] > 0.99
    
    eg = model.predict_dense(tf)
    assert eg.scores[1] > 0.99

    eg = model.predict_dense(ft)
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


@pytest.fixture
def bias_data():
    ff = np.asarray([0.,0.], dtype='f')
    costs0 = np.asarray([0., 1.], dtype='f')
    costs1 = np.asarray([1., 0.], dtype='f')
    def gen_random():
        if random.random() > 0.7:
            yield (ff, 0, costs0)
        else:
            yield (ff, 1, costs1)
    return gen_random


def test_linear_bias(bias_data):
    '''Test that a linear model can learn a bias.'''
    model = NeuralNet((2, 2), rho=0.0, eta=0.1, eps=1e-4, update_step='sgd')

    assert model.nr_in == 2
    assert model.nr_class == 2
    assert model.nr_layer == 2
    
    if not model.use_batch_norm:
        bias0, bias1 = model.weights[-2:]
    else:
        bias0, bias1 = model.weights[-4:-2]

    assert bias0 == 0
    assert bias1 == 0
    for _ in range(100):
        for feats, label, costs in bias_data():
            eg = model.train_dense(feats, costs)
    if not model.use_batch_norm:
        bias0, bias1 = model.weights[-2:]
    else:
        bias0, bias1 = model.weights[-4:-2]
    assert bias1 > bias0
    acc = 0.0
    total = 0
    for i in range(20):
        for features, label, costs in bias_data():
            eg = model.predict_dense(features)
            assert costs[label] == 0
            acc += eg.scores[label] > 0.5
            total += 1
    assert (acc/total) > 0.5
    assert (acc/total) < 1.0


def test_deep_bias(bias_data):
    '''Test that a deep model can learn a bias.'''
    model = NeuralNet((2,2,2,2,2,2,2, 2), rho=0.0, eta=0.1, eps=1e-4, update_step='adadelta')

    assert model.nr_in == 2
    assert model.nr_class == 2
    assert model.nr_layer > 2
    
    if not model.use_batch_norm:
        bias0, bias1 = model.weights[-2:]
    else:
        bias0, bias1 = model.weights[-4:-2]
    assert bias0 == 0
    assert bias1 == 0
    for _ in range(20):
        for feats, label, costs in bias_data():
            eg = model.train_dense(feats, costs)
    if not model.use_batch_norm:
        bias0, bias1 = model.weights[-2:]
    else:
        bias0, bias1 = model.weights[-4:-2]
 
    assert bias1 > bias0
    acc = 0.0
    total = 0
    for i in range(20):
        for features, label, costs in bias_data():
            eg = model.predict_dense(features)
            assert costs[label] == 0
            acc += eg.scores[label] > 0.5
            total += 1
    assert (acc/total) > 0.5
    assert (acc/total) < 1.0


def test_learn_linear(or_data):
    '''Test that a linear model can learn OR.'''
    # Need high eta on this sort of toy problem, or learning takes forever!
    model = NeuralNet((2, 2), rho=0.0, eta=0.1, eps=1e-4, update_step='sgd',
                       alpha=0.8)

    assert model.nr_in == 2
    assert model.nr_class == 2
    assert model.nr_layer == 2
    
    # It takes about this many iterations, with the settings above.
    for _ in range(900):
        for feats, label, costs in or_data:
            eg = model.train_dense(feats, costs)
        random.shuffle(or_data)
    for avg in model.averages:
        print(avg)
    acc = 0.0
    for features, label, costs in or_data:
        eg = model.predict_dense(features)
        assert costs[label] == 0
        acc += eg.scores[label] > 0.5
    assert acc == len(or_data)


def test_mlp_learn_linear(or_data):
    '''Test that with a hidden layer, we can still learn OR'''
    # Need high eta on this sort of toy problem, or learning takes forever!
    model = NeuralNet((2, 3, 2), rho=0.0, eta=0.5, eps=1e-4,
                      update_step='sgd')

    assert model.nr_in == 2
    assert model.nr_class == 2
    assert model.nr_layer == 3
    
    # Keep this set low, so that we see that the hidden layer allows the function
    # to be learned faster than the linear model
    for _ in range(50):
        for feats, label, costs in or_data:
            batch = model.train_dense(feats, costs)
        random.shuffle(or_data)
    acc = 0.0
    for features, label, costs in or_data:
        eg = model.predict_dense(features)
        assert costs[label] == 0
        acc += eg.scores[label] > 0.5
    assert acc == len(or_data)


def test_xor_gradient(xor_data):
    '''Test that after each update, we move towards the correct label.'''
    model = NeuralNet((2, 2, 2), rho=0.0, eta=0.1, update_step='adadelta')
    assert model.nr_in == 2
    assert model.nr_class == 2
    assert model.nr_layer == 3
    
    for _ in range(500):
        for i, (features, label, costs) in enumerate(xor_data):
            prev = list(model.predict_dense(features).scores)
            assert_allclose([sum(prev)], [1.0])
            tmp = model.train_dense(features, costs)
            eg = model.predict_dense(features)
            assert (prev[label] <= eg.scores[label] or \
                    prev[label] == eg.scores[label] == 1.0)


@pytest.mark.xfail
def test_xor_eta(xor_data):
    '''Test that a higher learning rate causes loss to decrease faster.'''
    small_eta_model = NeuralNet((2, 10,2), rho=0.0, eta=0.000001, update_step='sgd')
    normal_eta_model = NeuralNet((2, 10,2), rho=0.0, eta=0.1, update_step='sgd')
    small_eta_loss = 0.0
    normal_eta_loss = 0.0
    for _ in range(1000):
        for i, (features, label, costs) in enumerate(xor_data):
            small_eta_loss += small_eta_model.train_dense(features, costs).loss
            normal_eta_loss += normal_eta_model.train_dense(features, costs).loss
    assert normal_eta_loss < small_eta_loss


def test_xor_rho(xor_data):
    '''Test that higher L2 penalty causes slower learning.'''
    big_rho_model = NeuralNet((2,10,10,10,2), rho=0.8, eta=0.005, update_step='adadelta')
    normal_rho_model = NeuralNet((2, 10,10,10, 2), rho=1e-4, eta=0.005, update_step='adadelta')
    big_rho_model.weights = list(normal_rho_model.weights)
    big_rho_loss = 0.0
    normal_rho_loss = 0.0
    for _ in range(10):
        for i, (features, label, costs) in enumerate(xor_data):
            big_rho_loss += big_rho_model.train_dense(features, costs).loss
            normal_rho_loss += normal_rho_model.train_dense(features, costs).loss
    assert normal_rho_loss < (big_rho_loss * 1.1)


def test_xor_deep(xor_data):
    '''Compare 0, 1 and 3 layer networks.
    The 3 layer seems to do better, but it doesn't *have* to. But if the
    0 layer works, something's wrong!'''
    linear = NeuralNet((2,2), rho=0.0000, eta=0.1, update_step='sgd')
    small = NeuralNet((2,2,2), rho=0.0000, eta=0.1, update_step='sgd')
    big = NeuralNet((2,2,2,2,2,2), rho=0.0000, eta=0.1, update_step='sgd')
    for _ in range(1000):
        for i, (features, label, costs) in enumerate(xor_data):
            ln = linear.train_dense(features, costs)
            bg = big.train_dense(features, costs)
            sm = small.train_dense(features, costs)
        random.shuffle(xor_data)

    linear_loss = 0.0
    small_loss = 0.0
    big_loss = 0.0
    for i, (features, label, costs) in enumerate(xor_data):
        linear_loss += 1 - linear.predict_dense(features).scores[label]
        small_loss += 1 - small.predict_dense(features).scores[label]
        big_loss += 1 - big.predict_dense(features).scores[label]
    assert big_loss < 0.5
    assert small_loss < 2.0
    assert linear_loss > 1.9
 

def test_model_widths_sgd(or_data):
    '''Test different model widths'''
    narrow = NeuralNet((2,4,2), rho=0.0, eta=0.01, update_step='sgd')
    wide = NeuralNet((2,20,2), rho=0.0, eta=0.01, update_step='sgd')
    assert wide.nr_weight > narrow.nr_weight
    narrow_loss = 0.0
    wide_loss = 0.0
    for _ in range(20):
        for i, (features, label, costs) in enumerate(or_data):
            narrow_loss += narrow.train_dense(features, costs).loss
            wide_loss += wide.train_dense(features, costs).loss
        random.shuffle(or_data)
    # We don't know that the extra width is better, but it shouldn't be
    # *much* worse
    assert wide_loss < (narrow_loss * 1.1)
    # It also shouldn't be the same!
    assert wide_loss != narrow_loss


def test_embedding():
    model = NeuralNet((10,4,2), embed=((5,), (0,0)), rho=0.0, eta=0.005)
    assert model.nr_in == 10
    eg = model.Example({(0, 1): 2.5})
    model.predict_example(eg)
    assert eg.activation(0, 0) != 0
    assert eg.activation(0, 1) != 0
    assert eg.activation(0, 2) != 0
    assert eg.activation(0, 3) != 0
    assert eg.activation(0, 4) != 0
    
    eg = model.Example({(1, 1867): 0.5})
    model.predict_example(eg)
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
    

def test_sparse_backprop_single():
    model = NeuralNet((2, 2, 2), embed=((2,), (0,)), update_step='sgd', eta=0.1)
    x = {(0, 1): 4.0}
    y = (0, 1)
    b1 = model.train_sparse(x, y)
    b2 = model.train_sparse(x, y)
    b3 = model.train_sparse(x, y)
    b4 = model.train_sparse(x, y)
    b5 = model.train_sparse(x, y)
    assert b2.loss > b3.loss or b2.loss == 0.0
    assert b3.loss > b4.loss or b3.loss == 0.0
    assert b4.loss > b5.loss or b4.loss == 0.0


def f2s(fs):
    return ' '.join('%.3f' % val for val in fs)


def test_sparse_backprop():
    def train_batch(model, xy):
        x, y = xy
        loss = 0.0
        for x,y in zip(X, Y):
            eg = model.train_sparse(x, y)
            loss += eg.loss
        return loss

    model = NeuralNet((10, 2, 2), embed=((10,), (0,)), rho=0.0, eta=0.005,
                      update_step='sgd')
    X = [{(0, 1): 4.0, (0, 2): 3.0, (0, 3): 4.0, (0, 100): 1.0}, {(0, 10): 3.0,
         (0, 2): 2.0}]
    Y = [(0, 1), (1, 0)]
    b1 = train_batch(model, (X, Y))
    b2 = train_batch(model, (X, Y))
    b3 = train_batch(model, (X, Y))
    b4 = train_batch(model, (X, Y))
    b5 = train_batch(model, (X, Y))
    b6 = train_batch(model, (X, Y))
    b7 = train_batch(model, (X, Y))
    b8 = train_batch(model, (X, Y))
    b9 = train_batch(model, (X, Y))
    b10 = train_batch(model, (X, Y))

    assert b1 > b3 > b10
