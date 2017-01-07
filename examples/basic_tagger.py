from thinc.neural.id2vec import Embed
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


#class EncodeTagger(Network):
#    Input = WindowEncode
#    width = 32
#
#    def setup(self, nr_class, *args, **kwargs):
#        self.layers.append(
#            self.Input(vectors={}, W=None, nr_out=self.width,
#                nr_in=self.width, static=False))
#        self.layers.append(
#            ReLu(nr_out=self.width, nr_in=self.layers[-1].nr_out))
#        self.layers.append(
#            ReLu(nr_out=self.width, nr_in=self.layers[-1].nr_out))
#        self.layers.append(
#            Softmax(nr_out=nr_class, nr_in=self.layers[-1].nr_out))
#        self.set_weights(initialize=True)
#        self.set_gradient()
#

class EmbedTagger(Model):
    def __init__(self, nr_tag, width, vector_length, vectors=None):
        vectors = {} if vectors is None else vectors
        self.width = width
        self.vector_length = vector_length
        layers = [
            Embed(width, vector_length, vectors=vectors, ops=NumpyOps(),
                name='embed'),
            ExtractWindow(n=1, ops=NumpyOps(), name='extract'),
            ReLu(width, width*3, ops=NumpyOps(), name='relu1'),
            ReLu(width, width, ops=NumpyOps(), name='relu2'),
            ExtractWindow(n=2, ops=NumpyOps(), name='extract2'),
            ReLu(width, width*5, ops=NumpyOps(), name='relu3'),
            ReLu(width, width, ops=NumpyOps(), name='relu4'),
            Softmax(nr_tag, width, ops=NumpyOps(), name='softmax')
        ]
        Model.__init__(self, *layers, ops=NumpyOps())

    def check_input(self, X, expect_batch=False):
        return True

    def add_vector(self, id_):
        self.layers[0].add_vector(id_, self.vector_length, add_gradient=True)


class CascadeTagger(Model):
    '''A silly example of using the gradient more flexibly.

    Here we train the EmbedTagger on the tagging objective, but use the results
    as features, and additionally backprop it with that objective.
    '''
    def __init__(self, nr_tag, width, vector_length, vectors=None):
        self.width = width
        self.nr_tag = nr_tag
        self.vector_length = vector_length
        self.vectors = vectors
        self.first = EmbedTagger(nr_tag, width, vector_length, vectors=vectors)
        self.embed2 = Embed(
            width, vector_length, vectors=vectors,
            ops=NumpyOps(), name='embed2.1')
        self.relu2 = ReLu(width, width + nr_tag, ops=NumpyOps(), name='relu2.1')
        self.relu3 = ReLu(width, width + nr_tag, name='relu2.2')
        self.softmax2 = Softmax(nr_tag, width + nr_tag, name='softmax2.1')
        layers = [self.first, self.embed2, self.relu2, self.relu3, self.softmax2]
        Model.__init__(self, *layers, ops=NumpyOps())

    def begin_update(self, X, **kwargs):
        first_tags, upd_tag1 = self.first.begin_update(X, **kwargs)
        X, upd_embed2 = self.embed2.begin_update(X, **kwargs)
        X = self.ops.xp.hstack([first_tags, X])
        X, upd_rel2 = self.relu2.begin_update(X, **kwargs)
        X = self.ops.xp.hstack([first_tags, X])
        X, upd_rel3 = self.relu3.begin_update(X, **kwargs)
        X = self.ops.xp.hstack([first_tags, X])
        X, upd_sm2 = self.softmax2.begin_update(X, **kwargs)

        def finish_update(gradient, sgd):
            def unstack(gradient):
                split = first_tags.shape[1]
                return gradient[:, :split], gradient[:, split:]

            upd_tag1(gradient, sgd, is_child=True)

            grad1, grad2 = unstack(upd_sm2(gradient, sgd, is_child=True))
            upd_tag1(grad1, sgd, is_child=True)

            grad1, grad2 = unstack(upd_rel3(grad2, sgd, is_child=True))
            upd_tag1(grad1, sgd, is_child=True)

            grad1, grad2 = unstack(upd_rel2(grad2, sgd, is_child=True))
            upd_tag1(grad1, sgd, is_child=True)
 
            upd_embed2(grad2, sgd, is_child=True)

            sgd(self.params.weights, self.params.gradient, key=('', self.name))
            return None
        return X, finish_update

    def add_vector(self, id_):
        self.first.add_vector(id_)
        self.embed2.add_vector(id_, self.vector_length, add_gradient=True)


def main():
    train_data, check_data, nr_class = ancora_pos_tags()
    model = EmbedTagger(nr_class, 32, 8, vectors={})
    for X, y in train_data:
        for x in X:
            model.add_vector(x)

    dev_X, dev_Y = zip(*check_data)
    dev_X = model.ops.flatten(dev_X)
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
            examples = model.ops.flatten(examples)
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
