from __future__ import print_function
from timeit import default_timer as timer
import plac
import numpy

from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, ReLu, Softmax
from thinc.neural._classes.convolution import ExtractWindow
from thinc.neural._classes.maxout import Maxout
from thinc.neural._classes.batchnorm import BatchNorm

from thinc.loss import categorical_crossentropy
from thinc.api import layerize, chain, clone
from thinc.neural.util import flatten_sequences, remap_ids
from thinc.neural.ops import NumpyOps, CupyOps

from thinc.extra.datasets import ancora_pos_tags

def to_categorical(y, nb_classes=None):
    # From keras
    y = numpy.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = numpy.max(y) + 1
    n = y.shape[0]
    categorical = numpy.zeros((n, nb_classes), dtype='float32')
    categorical[numpy.arange(n), y] = 1
    return categorical


def main(width=64, vector_length=64):
    train_data, check_data, nr_tag = ancora_pos_tags(numpy)

    #Model.ops = CupyOps()
    with Model.define_operators({'**': clone, '>>': chain}):
        model = (
            Embed(width, vector_length, nV=5000)
            >> ExtractWindow(nW=1)
            >> Maxout(300)
            >> ExtractWindow(nW=1)
            >> Maxout(300)
            >> ExtractWindow(nW=1)
            >> Maxout(300)
            >> Softmax(nr_tag))

    train_X, train_y = zip(*train_data)
    print("NR vector", max(max(seq) for seq in train_X))
    dev_X, dev_y = zip(*check_data)
    n_train = sum(len(x) for x in train_X)
    remapping = remap_ids(NumpyOps())
    train_X = remapping(flatten_sequences(train_X)[0])[0]
    dev_X = remapping(flatten_sequences(dev_X)[0])[0]
    train_y = flatten_sequences(train_y)[0]
    train_y = to_categorical(train_y, nb_classes=nr_tag)
    dev_y = flatten_sequences(dev_y)[0]
    dev_y = to_categorical(dev_y, nb_classes=nr_tag)
    train_X = model.ops.asarray(train_X)
    train_y = model.ops.asarray(train_y)
    dev_X = model.ops.asarray(dev_X)
    dev_y = model.ops.asarray(dev_y)
    with model.begin_training(train_X, train_y) as (trainer, optimizer):
        trainer.batch_size = 128
        trainer.nb_epoch = 20
        trainer.dropout = 0.0
        trainer.dropout_decay = 1e-4
        epoch_times = [timer()]
        def track_progress():
            start = timer()
            acc = model.evaluate(dev_X, dev_y)
            end = timer()
            with model.use_params(optimizer.averages):
                avg_acc = model.evaluate(dev_X, dev_y)
            stats = (
                acc,
                avg_acc,
                float(n_train) / (end-epoch_times[-1]),
                float(dev_y.shape[0]) / (end-start))
            print("%.3f (%.3f) acc, %d wps train, %d wps run" % stats)
            epoch_times.append(end)
        trainer.each_epoch.append(track_progress)
        for X, y in trainer.iterate(train_X, train_y):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            #d_loss, loss = categorical_crossentropy(yh, y)
            #optimizer.set_loss(loss)
            backprop(yh-y, optimizer)
    with model.use_params(optimizer.averages):
        print(model.evaluate(dev_X, dev_y))
 

if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
