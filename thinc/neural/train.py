# coding: utf8
from __future__ import unicode_literals

import numpy.random
from tqdm import tqdm

from .optimizers import Adam, linear_decay


class Trainer(object):
    def __init__(self, model, **cfg):
        self.ops = model.ops
        self.model = model
        self.L2 = cfg.get("L2", 0.0)
        self.optimizer = Adam(model.ops, 0.001, decay=0.0, eps=1e-8, L2=self.L2)
        self.batch_size = cfg.get("batch_size", 128)
        self.nb_epoch = cfg.get("nb_epoch", 20)
        self.i = 0
        self.dropout = cfg.get("dropout", 0.0)
        self.dropout_decay = cfg.get("dropout_decay", 0.0)
        self.each_epoch = []

    def __enter__(self):
        return self, self.optimizer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.use_params(self.optimizer.averages)

    def iterate(self, train_X, train_y, progress_bar=True):
        orig_dropout = self.dropout
        for i in range(self.nb_epoch):
            indices = numpy.arange(len(train_X))
            numpy.random.shuffle(indices)
            indices = self.ops.asarray(indices)
            j = 0
            with tqdm(total=indices.shape[0], leave=False) as pbar:
                while j < indices.shape[0]:
                    slice_ = indices[j : j + self.batch_size]
                    X = _take_slice(train_X, slice_)
                    y = _take_slice(train_y, slice_)
                    yield X, y
                    self.dropout = linear_decay(orig_dropout, self.dropout_decay, j)
                    j += self.batch_size
                    if progress_bar:
                        pbar.update(self.batch_size)
            for func in self.each_epoch:
                func()


def _take_slice(data, slice_):
    if isinstance(data, list) or isinstance(data, tuple):
        return [data[int(i)] for i in slice_]
    else:
        return data[slice_]
