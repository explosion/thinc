import plac
import torch
import numpy

from thinc.neural.optimizers import SGD

from torch import autograd
from torch import nn
import torch.optim

from thinc.extra.wrappers import PyTorchWrapperRNN

def test_rnn(
    rnn_type='LSTM', 
    input_size=500, 
    hidden_size=50, 
    sequence_length=10, 
    batch_size=8,
    num_layers=2,
    bidirectional=True,
    lr=0.001):

    rnn_model = getattr(nn, rnn_type)(
        input_size, 
        hidden_size, 
        num_layers, 
        bidirectional=bidirectional)

    num_directions = 2 if bidirectional else 1
    
    model = PyTorchWrapperRNN(rnn_model)
    sgd = SGD(model.ops, lr)

    # Inputs and expected ouput
    X, Y = generate_data(sequence_length, batch_size, input_size, hidden_size, num_directions)
    
    # Test output shape
    check_Y_shape(model, X, Y, sequence_length, batch_size, hidden_size, num_directions)

    # Test initializing hidden
    initial_hidden = init_hidden(rnn_model, rnn_type, batch_size, num_layers*num_directions, hidden_size)
    # Test if sum of rnn output converges to with initial hidden     
    check_learns_zero_output_rnn(model, sgd, X, Y, initial_hidden)

def check_Y_shape(model, X, Y, sequence_length, batch_size, hidden_size, num_directions):
    outputs, get_dX = model.begin_update(X)
    Yh, _ = outputs
    assert Yh.shape == (sequence_length, batch_size, hidden_size*num_directions)

def check_learns_zero_output_rnn(model, sgd, X, Y, initial_hidden=None):
    '''Check we can learn to output a zero vector'''
    outputs, get_dX = model.begin_update(X)
    Yh, h_n = outputs
    dYh = (Yh - Y) / len(Y)
    tupleDy = (dYh, 0)
    dX = get_dX(tupleDy, sgd=sgd)
    prev = numpy.abs(Yh.sum())
    print(prev)
    for i in range(1000):
        if(i == 0 and initial_hidden is not None):
            outputs, get_dX = model.begin_update(X, h_0=initial_hidden)
        else:
            outputs, get_dX = model.begin_update(X)
        Yh, h_n = outputs
        current_sum = numpy.abs(Yh.sum())
        tupleDy = (Yh-Y, h_n)
        dX = get_dX(tupleDy, sgd=sgd)

    # Should have decreased
    print(current_sum)

def generate_data(sequence_length, batch_size, input_size, hidden_size, num_directions):
    X = numpy.zeros((sequence_length, batch_size, input_size), dtype='f')
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    return X, torch.zeros(sequence_length, batch_size, hidden_size*num_directions)

def init_hidden(rnn_model, rnn_type, batch_size, num_layers, hidden_size):
        weight = next(rnn_model.parameters()).data
        if rnn_type == 'LSTM':
            return (autograd.Variable(weight.new(num_layers, batch_size, hidden_size).zero_()),
                    autograd.Variable(weight.new(num_layers, batch_size, hidden_size).zero_()))
        else:
            return autograd.Variable(weight.new(num_layers, batch_size, hidden_size).zero_())

if __name__ == '__main__':
    for rnn in ['RNN', 'GRU', 'LSTM']:
        for bi in [False, True]:
            test_title = "Using {bidirectional}{rnn_type}".format(
                rnn_type=rnn, 
                bidirectional="Bi-" if bi else "")
            print(test_title)
            test_rnn(rnn_type=rnn, bidirectional=bi)