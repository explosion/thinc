from __future__ import print_function
from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, ReLu, Softmax
from thinc.neural.vecs2vecs import ExtractWindow

from thinc.neural.util import score_model
from thinc.neural.optimizers import linear_decay
from thinc.neural.ops import NumpyOps
from thinc.loss import categorical_crossentropy
from thinc.api import layerize

from thinc.extra.datasets import ancora_pos_tags

import plac

try:
    import cytoolz as toolz
except ImportError:
    import toolz


def main():
    train_data, check_data, nr_class = ancora_pos_tags()

    with Model.define_operators({'>>': chain}):
        model = (
                  mark_sentence_boundaries
                  >> flatten_sentences
                  >> Embed(width, vector_length)
                  >> ReLu(width)
                  >> ReLu(width)
                  >> Softmax(nr_tag)
                )

    dev_X, dev_Y = zip(*check_data)
    dev_Y = model.ops.flatten(dev_Y)
    with model.begin_training(train_data) as (trainer, optimizer):
        trainer.batch_size = 8
        trainer.nb_epoch = 10
        trainer.dropout = 0.3
        trainer.dropout_decay = 0.
        trainer.nb_epoch = 10
        trainer.each_epoch(lambda: print(model.evaluate(dev_X, dev_y)))
        for X, y in trainer.iterate(train_X, train_y):
            y = model.ops.flatten(y)
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            d_loss, loss = categorical_crossentropy(yh, y)
            optimizer.set_loss(loss)
            backprop(d_loss, optimizer)
    with model.use_params(optimizer.averages):
        print('Avg dev.: %.3f' % score_model(model, dev_X, dev_Y))
 

if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
