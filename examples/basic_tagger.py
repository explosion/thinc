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


try:
    import cytoolz as toolz
except ImportError:
    import toolz


@toolz.curry
def flatten(ops, X, dropout=0.0):
    def finish_update(grad, *args, **kwargs):
        return grad
    return ops.flatten(X), finish_update


class EmbedTagger(Model):
    def __init__(self, nr_tag, width, vector_length, vectors=None):
        vectors = {} if vectors is None else vectors
        self.width = width
        self.vector_length = vector_length
        layers = [
            layerize(flatten(NumpyOps())),
            Embed(width, vector_length, vectors=vectors, ops=NumpyOps(),
                name='embed'),
            ReLu(width, width, ops=NumpyOps(), name='relu1'),
            ReLu(width, width, ops=NumpyOps(), name='relu2'),
            Softmax(nr_tag, width, ops=NumpyOps(), name='softmax')
        ]
        Model.__init__(self, *layers, ops=NumpyOps())


def main():
    train_data, check_data, nr_class = ancora_pos_tags()
    model = EmbedTagger(nr_class, 32, 8, vectors={})

    dev_X, dev_Y = zip(*check_data)
    dev_Y = model.ops.flatten(dev_Y)
    with model.begin_training(train_data) as (trainer, optimizer):
        trainer.nb_epoch = 10
        trainer.batch_size = 8
        trainer.nb_epoch = 10
        trainer.dropout = 0.3
        trainer.dropout_decay = 0.
        for examples, truth in trainer.iterate(model, train_data, dev_X, dev_Y,
                                               nb_epoch=trainer.nb_epoch):
            truth = model.ops.flatten(truth)
            guess, finish_update = model.begin_update(examples,
                                        dropout=trainer.dropout)

            gradient, loss = categorical_crossentropy(guess, truth)
            optimizer.set_loss(loss)
            finish_update(gradient, optimizer)
            trainer._loss += loss / len(truth)
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
