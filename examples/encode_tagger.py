from thinc.neural.id2vec import Embed
from thinc.neural.ids2vecs import MaxoutWindowEncode
from thinc.neural.vec2vec import Model, ReLu, Softmax
from thinc.neural.vecs2vecs import ExtractWindow

from thinc.neural.util import score_model
from thinc.neural.optimizers import linear_decay
from thinc.neural.ops import NumpyOps
from thinc.loss import categorical_crossentropy

from thinc.extra.datasets import ancora_pos_tags

import plac

try:
    import cytoolz as toolz
except ImportError:
    import toolz


class EmbedEncode(Model):
    @property
    def embed(self):
        return self.layers[0]
    @property
    def encode(self):
        return self.layers[1] if len(self.layers) >= 2 else None
    @property
    def nr_vector(self):
        return len(self.layers[0].vectors)

    def predict_batch(self, ids):
        flat_ids = self.ops.flatten(ids)
        lengths = [len(seq) for seq in ids]

        vectors = self.embed.predict_batch(flat_ids)
        if self.encode:
            return self.encode.predict_batch((flat_ids, vectors, lengths))
        else:
            return vectors

    def begin_update(self, ids, dropout=0.0):
        flat_ids = self.ops.flatten(ids)
        self._insert_ids(flat_ids)
        lengths = [len(seq) for seq in ids]
        vectors, bp_embed = self.embed.begin_update(flat_ids)
        if self.encode is None:
            return vectors, bp_embed

        vectors, bp_embed = self.embed.begin_update(flat_ids, dropout=dropout)

        output, bp_encode = self.encode.begin_update((flat_ids, vectors, lengths),
                                dropout=dropout)

        def finish_update(gradient, optimizer=None, **kwargs):
            gradient = bp_encode(gradient, optimizer, **kwargs)
            gradient = bp_embed(gradient, optimizer, **kwargs)
            return gradient

        return output, finish_update

    def _insert_ids(self, ids):
        for id_ in ids:
            vector = self.embed.get_vector(id_)
            if vector is None:
                self.embed.add_vector(id_, self.input_shape, add_gradient=True)


class Tagger(Model):
    @property
    def nr_vector(self):
        return self.layers[0].nr_vector

    def __init__(self, nr_tag, width, vector_length, vectors=None):
        vectors = {} if vectors is None else vectors
        self.width = width
        self.vector_length = vector_length
        layers = [
            EmbedEncode(
                Embed(vector_length, vector_length, vectors=vectors, ops=NumpyOps(),
                    name='embed'),
                MaxoutWindowEncode(width, nr_in=vector_length, ops=NumpyOps(),
                    name='encode'),
                name='embedencode'
            ),
            ExtractWindow(n=2),
            ReLu(width, width*5, ops=NumpyOps(), name='relu1'),
            #ReLu(width, width, ops=NumpyOps(), name='relu2'),
            #ExtractWindow(n=4),
            #ReLu(width, width*9, ops=NumpyOps(), name='relu3'),
            #ReLu(width, width, ops=NumpyOps(), name='relu4'),
            Softmax(nr_tag, width, ops=NumpyOps(), name='softmax')
        ]
        Model.__init__(self, *layers, ops=NumpyOps())

    def check_input(self, X, expect_batch=False):
        return True

    def add_vector(self, id_):
        self.layers[0].add_vector(id_, self.vector_length, add_gradient=True)


def main():
    train_data, check_data, nr_class = ancora_pos_tags()
    model = Tagger(nr_class, 32, 32, vectors={})

    dev_X, dev_Y = zip(*check_data)
    dev_Y = model.ops.flatten(dev_Y)
    with model.begin_training(train_data) as (trainer, optimizer):
        trainer.batch_size = 8
        trainer.nb_epoch = 10
        trainer.dropout = 0.0
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
        cProfile.run("plac.call(main)", "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time", "cumulative").print_stats(100)
        s.strip_dirs().sort_stats('ncalls').print_callers()
