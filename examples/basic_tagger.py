from __future__ import print_function
from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, ReLu, Softmax
from thinc.neural._classes.feed_forward import FeedForward

from thinc.neural.ops import NumpyOps
from thinc.loss import categorical_crossentropy
from thinc.api import layerize, chain
from thinc.neural.util import flatten_sequences

from thinc.extra.datasets import ancora_pos_tags

import plac

try:
    import cytoolz as toolz
except ImportError:
    import toolz


def main(width=128, vector_length=64):
    train_data, check_data, nr_tag = ancora_pos_tags()

    model = FeedForward((
              layerize(flatten_sequences),
              Embed(width, vector_length),
              ReLu(width),
              ReLu(width),
              Softmax(nr_tag)))

    train_X, train_y = zip(*train_data)
    dev_X, dev_y = zip(*check_data)
    dev_y = model.ops.flatten(dev_y)
    with model.begin_training(train_X, train_y) as (trainer, optimizer):
        trainer.batch_size = 8
        trainer.nb_epoch = 10
        trainer.dropout = 0.3
        trainer.dropout_decay = 0.
        trainer.nb_epoch = 10
        trainer.each_epoch(lambda: print(model._layers[1].vectors[1]))
        trainer.each_epoch(lambda: print(model.evaluate(dev_X, dev_y)))
        for X, y in trainer.iterate(train_X, train_y):
            y = model.ops.flatten(y)
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            d_loss, loss = categorical_crossentropy(yh, y)
            optimizer.set_loss(loss)
            backprop(d_loss, optimizer)
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
