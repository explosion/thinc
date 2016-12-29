from thinc.datasets import conll_pos_tags
from thinc.base import Network
from thinc.id2vec import Embed
from thinc.vec2vec import ReLu
from thinc.vec2vec import Softmax
from thinc.convolution import ExtractWindow
from thinc.ids2vecs import WindowEncode

from thinc.util import score_model
from thinc.optimizers import linear_decay

import plac

try:
    import cytoolz as toolz
except ImportError:
    import toolz


class EncodeTagger(Network):
    Input = WindowEncode
    width = 32

    def setup(self, nr_class, *args, **kwargs):
        self.layers.append(
            self.Input(vectors={}, W=None, nr_out=self.width,
                nr_in=self.width, static=False))
        self.layers.append(
            ReLu(nr_out=self.width, nr_in=self.layers[-1].nr_out))
        self.layers.append(
            ReLu(nr_out=self.width, nr_in=self.layers[-1].nr_out))
        self.layers.append(
            Softmax(nr_out=nr_class, nr_in=self.layers[-1].nr_out))
        self.set_weights(initialize=True)
        self.set_gradient()


class EmbedTagger(EncodeTagger):
    class Input(Network):
        nr_in = None
        nr_out = None
        def setup(self, *args, **kwargs):
            self.layers = [
                Embed(vectors={}, W=None, nr_in=self.nr_in, nr_out=self.nr_out),
                ExtractWindow(n=1, nr_in=self.nr_in, nr_out=self.nr_out*3)
            ]
            self.nr_out *= 3


def _flatten(ops, data):
    X, y = zip(*data)
    X = ops.asarray(list(toolz.concat(X)), dtype='i')
    y = ops.asarray(list(toolz.concat(y)), dtype='i')
    return X, y


def main():
    train_data, check_data, nr_class = conll_pos_tags()
    model = EncodeTagger(nr_class)
    for X, y in train_data:
        for x in X:
            model.layers[0].add_vector(x,
                model.width, add_gradient=True)

    with model.begin_training(train_data) as (trainer, optimizer):
        trainer.nb_epoch = 10
        for examples, truth in trainer.iterate(model, train_data, check_data,
                                               nb_epoch=trainer.nb_epoch):
            truth = model.ops.flatten(truth)
            guess, finish_update = model.begin_update(examples, dropout=0.0)
            gradient, loss = trainer.get_gradient(guess, truth)
            optimizer.set_loss(loss)
            finish_update(gradient, optimizer)


if __name__ == '__main__':
    plac.call(main)
