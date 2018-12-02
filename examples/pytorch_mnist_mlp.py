from __future__ import print_function
import plac
from srsly import cloudpickle as pickle
from tqdm import tqdm
from thinc.v2v import Model, ReLu, Softmax
from thinc.api import clone, chain
from thinc.neural.util import to_categorical, prefer_gpu

from thinc.extra import datasets
from thinc.neural.ops import CupyOps
from thinc.extra.wrappers import PyTorchWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyTorchFeedForward(nn.Module):
    def __init__(self, depth, width, input_size, output_size):
        super(PyTorchFeedForward, self).__init__()
        self.linears = [nn.Linear(input_size, width)]
        for i in range(depth-1):
            self.linears.append(nn.Linear(width, width))
        self.linears.append(nn.Linear(width, output_size))
        for i, child in enumerate(self.linears):
            self.add_module('child%d' % i, child)

    def forward(self, x):
        y = F.dropout(F.relu(self.linears[0](x)), self.training)
        for layer in self.linears[1:-1]:
            y = F.relu(layer(y))
            y = F.dropout(y, self.training)
        y = F.log_softmax(self.linears[-1](y))
        return y


def main(depth=2, width=512, nb_epoch=30):
    prefer_gpu()
    torch.set_num_threads(1)

    train_data, dev_data, _ = datasets.mnist()
    train_X, train_y = Model.ops.unzip(train_data)
    dev_X, dev_y = Model.ops.unzip(dev_data)

    dev_y = to_categorical(dev_y)
    model = PyTorchWrapper(
        PyTorchFeedForward(depth=depth, width=width, input_size=train_X.shape[1],
            output_size=dev_y.shape[1]))
    with model.begin_training(train_X, train_y, L2=1e-6) as (trainer, optimizer):
        epoch_loss = [0.]
        def report_progress():
            #with model.use_params(optimizer.averages):
            print(epoch_loss[-1], model.evaluate(dev_X, dev_y), trainer.dropout)
            epoch_loss.append(0.)

        trainer.each_epoch.append(report_progress)
        trainer.nb_epoch = nb_epoch
        trainer.dropout = 0.3
        trainer.batch_size = 128
        trainer.dropout_decay = 0.0
        train_X = model.ops.asarray(train_X, dtype='float32')
        y_onehot = to_categorical(train_y)
        for X, y in trainer.iterate(train_X, y_onehot):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            loss = ((yh-y)**2.).sum() / y.shape[0]
            backprop(yh-y, optimizer)
            epoch_loss[-1] += loss
        with model.use_params(optimizer.averages):
            print('Avg dev.: %.3f' % model.evaluate(dev_X, dev_y))
            with open('out.pickle', 'wb') as file_:
                pickle.dump(model, file_, -1)


if __name__ == '__main__':
    plac.call(main)
