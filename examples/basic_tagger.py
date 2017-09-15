from __future__ import print_function
from thinc.neural._classes.hash_embed import HashEmbed
from thinc.neural.vec2vec import Model, ReLu, Softmax

from thinc.api import layerize, chain, with_flatten

from thinc.extra.datasets import ancora_pos_tags
from thinc.neural.util import to_categorical

import plac

try:
    import cytoolz as toolz
except ImportError:
    import toolz


def main(width=32, nr_vector=1000):
    train_data, check_data, nr_tag = ancora_pos_tags(encode_words=True)

    model = with_flatten(
                 chain(
                    HashEmbed(width, 1000),
                    Softmax(nr_tag, width)))

    train_X, train_y = zip(*train_data)
    dev_X, dev_y = zip(*check_data)
    train_y = [to_categorical(y, nb_classes=nr_tag) for y in train_y]
    dev_y = [to_categorical(y, nb_classes=nr_tag) for y in dev_y]
    with model.begin_training(train_X, train_y) as (trainer, optimizer):
        trainer.each_epoch.append(
            lambda: print(model.evaluate(dev_X, dev_y)))
        for X, y in trainer.iterate(train_X, train_y):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            backprop([yh[i]-y[i] for i in range(len(yh))], optimizer)
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
