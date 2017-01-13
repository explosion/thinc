from __future__ import unicode_literals, print_function

from .optimizers import Eve, Adam, linear_decay
from .util import minibatch

import numpy.random
import tqdm


class Trainer(object):
    def __init__(self, model, train_data, L2=0.0):
        self.ops = model.ops
        self.model = model
        self.optimizer = Eve(model.ops, 0.001)
        self.batch_size = 128
        self.nb_epoch = 1
        self.i = 0
        self.L2 = 0.0
        self.dropout = 0.9
        self.dropout_decay = 1e-4
        self._loss = 0.

    def __enter__(self):
        return self, self.optimizer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.use_params(self.optimizer.averages)

    def each_epoch(self, func):
        pass

    def iterate(self, train_X, train_y):
        orig_dropout = self.dropout
        for i in range(self.nb_epoch):
            indices = self.ops.xp.asarray(range(len(train_X)))
            numpy.random.shuffle(indices)
            j = 0
            while j < len(indices):
                slice_ = indices[j : j + self.batch_size]
                X = train_X[slice_]
                y = train_y[slice_]
                yield X, y
                self.dropout = linear_decay(orig_dropout, self.dropout_decay,
                                            self.optimizer.nr_iter)
                j += self.batch_size
