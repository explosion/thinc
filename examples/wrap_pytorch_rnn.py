import plac
import torch

from torch import autograd
from torch import nn
import torch.optim

from thinc.extra.wrappers import PyTorchWrapperRNN

def main(rnn_type='GRU', 
    input_size=50, 
    hidden_size=25, 
    sequence_length=10, 
    batch_size=16,
    num_layers=1,
    bidirectional=False):

    rnn_model = getattr(nn, rnn_type)(
        input_size, 
        hidden_size, 
        num_layers, 
        bidirectional=bidirectional)
    
    optimizer = torch.optim.Adam(rnn_model.parameters())

    model = PyTorchWrapperRNN(rnn_model)

    # input RNN shape (seq_len, batch_size, input_size)
    X = torch.ones(sequence_length, batch_size, input_size)

    for i in range(10):
        # TODO: We can pass h_0 to begin_update
        yh, get_dX = model.begin_update(X, sgd=optimizer)
        # Calculate dY and dX using get_dX callback

if __name__ == '__main__':
    plac.call(main)