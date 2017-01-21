from __future__ import print_function
from timeit import default_timer as timer

from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, ReLu, Softmax
from thinc.neural._classes.feed_forward import FeedForward
from thinc.neural._classes.batchnorm import BatchNorm as BN
from thinc.neural._classes.convolution import ExtractWindow
from thinc.neural._classes.window_encode import MaxoutWindowEncode

from thinc.neural.ops import NumpyOps
from thinc.loss import categorical_crossentropy
from thinc.api import layerize, chain, clone
from thinc.neural.util import flatten_sequences

from thinc.extra.datasets import ancora_pos_tags

import plac


try:
    import cytoolz as toolz
except ImportError:
    import toolz


def get_positions(ids, drop=0.):
    positions = {id_: [] for id_ in set(ids)}
    for i, id_ in enumerate(ids):
        positions[id_].append(i)
    return positions, None


def main(width=64, vector_length=64):
    train_data, check_data, nr_tag = ancora_pos_tags()

    with Model.define_operators({'**': clone, '>>': chain}):
        #model = (
        #    layerize(flatten_sequences)
        #    >> layerize(get_positions)
        #    >> MaxoutWindowEncode(Embed(width, vector_length), 128,
        #          pieces=2, window=2)
        #    >> ExtractWindow(nW=2)
        #    >> ReLu(300)
        #    >> Softmax(nr_tag))
        #model = (
        #    layerize(flatten_sequences)
        #    >> Embed(width, vector_length)
        #    >> ExtractWindow(nW=2)
        #    >> ReLu(300)
        #    >> Softmax(nr_tag))
        model = (
            layerize(flatten_sequences)
            >> layerize(get_positions)
            >> MaxoutWindowEncode(Embed(width, vector_length), 300,
                  pieces=2, window=2)
            >> Softmax(nr_tag))

    train_X, train_y = zip(*train_data)
    print("NR vector", max(max(seq) for seq in train_X))
    dev_X, dev_y = zip(*check_data)
    dev_y = model.ops.flatten(dev_y)
    with model.begin_training(train_X, train_y) as (trainer, optimizer):
        trainer.batch_size = 16
        trainer.nb_epoch = 3
        trainer.dropout = 0.0
        trainer.dropout_decay = 0.
        epoch_times = [timer()]
        def track_progress():
            start = timer()
            acc = model.evaluate(dev_X, dev_y)
            end = timer()
            stats = (acc, end-epoch_times[-1], float(dev_y.shape[0]) / (end-start))
            print("%.3f acc, %d sec train, %d wps run" % stats)
            epoch_times.append(end)
        trainer.each_epoch.append(track_progress)
        for X, y in trainer.iterate(train_X, train_y):
            y = model.ops.flatten(y)
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            d_loss, loss = categorical_crossentropy(yh, y)
            optimizer.set_loss(loss)
            backprop(d_loss, optimizer)
    with model.use_params(optimizer.averages):
        print(model.evaluate(dev_X, dev_y))
 

if __name__ == '__main__':
    if 0:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
