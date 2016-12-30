from __future__ import print_function, unicode_literals, division
from thinc.datasets import conll_pos_tags
from thinc.base import Network
from thinc.id2vec import Embed
from thinc.vec2vec import ReLu, Maxout
from thinc.vec2vec import Softmax
from thinc.convolution import ExtractWindow
from thinc.doc2vecs import SpacyWindowEncode

from thinc.util import score_model
from thinc.optimizers import linear_decay
from thinc.datasets import read_conll
import spacy
from spacy.attrs import SHAPE
from spacy.tokens import Doc
import spacy.orth
import pathlib

import plac

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import cytoolz as toolz
except ImportError:
    import toolz


class SumTokens(Network):
    @property
    def nr_out(self):
        return self.embed.nr_out

    @property
    def nr_in(self):
        return self.embed.nr_in
    
    def setup(self, encode, embed):
        self.encode = encode
        self.embed = embed
        self.layers = [self.embed, self.encode]
 
    def predict_batch(self, X):
        encoded = self.encode.predict_batch(X)
        features = self.ops.flatten([(w.orth if w.rank < 5000 else w.shape)
                                      for w in doc] for doc in X)
        return encoded + self.embed.predict_batch(self.ops.asarray(features, dtype='i'))

    def begin_update(self, X, dropout=0.0):
        encoded, finish_encode = self.encode.begin_update(X, dropout=dropout)
        features = self.ops.flatten([(w.orth if w.rank < 5000 else w.shape)
                                      for w in doc] for doc in X)
        features = self.ops.asarray(features, dtype='i')
        embeded, finish_embed = self.embed.begin_update(features, dropout=dropout)
        def finish_update(gradients, optimizer=None, **kwargs):
            finish_encode(gradients, optimizer=optimizer, **kwargs)
            finish_embed(gradients, optimizer=optimizer, **kwargs)
            return gradients
        return encoded + embeded, finish_update


class ConcatTokens(SumTokens):
    @property
    def nr_out(self):
        return self.embed.nr_out + self.encode.nr_out

    def predict_batch(self, X):
        encoded = self.encode.predict_batch(X)
        #features = self.ops.flatten([w.shape for w in doc] for doc in X)
        features = self.ops.flatten([(w.orth if w.rank < 1000 else 0)
                                      for w in doc] for doc in X)
        embeded = self.embed.predict_batch(self.ops.asarray(features, dtype='i'))
        return self.ops.xp.hstack((embeded, encoded))

    def begin_update(self, X, dropout=0.0):
        #features = self.ops.flatten([w.shape for w in doc] for doc in X)
        features = self.ops.flatten([(w.orth if w.rank < 1000 else 0)
                                      for w in doc] for doc in X)
        features = self.ops.asarray(features, dtype='i')
        embeded, finish_embed = self.embed.begin_update(features, dropout=dropout)
        encoded, finish_encode = self.encode.begin_update(X, dropout=dropout)
        def finish_update(gradients, optimizer=None, **kwargs):
            embed_grad = gradients[:, :self.embed.nr_out]
            encode_grad = gradients[:, self.embed.nr_out:]
            finish_encode(encode_grad, optimizer=optimizer, **kwargs)
            finish_embed(embed_grad, optimizer=optimizer, **kwargs)
            return gradients
        return self.ops.xp.hstack((embeded, encoded)), finish_update


class EncodeTagger(Network):
    width = 128
    maxout_pieces = 3
    nr_in = None
    
    def setup(self, nr_class, *args, **kwargs):
        self.layers.append(
            ConcatTokens(
                SpacyWindowEncode(
                    vectors={}, W=None, nr_out=self.width,
                    nr_in=self.nr_in, nr_piece=self.maxout_pieces),
                Embed(
                    vectors={}, W=None, nr_out=self.width // 2, nr_in=self.width // 4))
        )
        self.layers.append(
            ReLu(nr_out=128, nr_in=self.layers[-1].nr_out))
        self.layers.append(
            ReLu(nr_out=128, nr_in=self.layers[-1].nr_out))
        self.layers.append(
            Softmax(nr_out=nr_class, nr_in=self.layers[-1].nr_out))
        self.set_weights(initialize=True)
        self.set_gradient()


def spacy_conll_pos_tags(nlp, train_loc, dev_loc):
    train_sents = list(read_conll(train_loc))
    dev_sents = list(read_conll(dev_loc))
    tagmap = {}
    for words, tags, heads, labels in train_sents:
        for tag in tags:
            tagmap.setdefault(tag, len(tagmap))
    def _encode(sents):
        X = []
        y = []
        oovs = 0
        n = 0
        for words, tags, heads, labels in sents:
            X.append(Doc(nlp.vocab, words=words))
            y.append([tagmap[tag] for tag in tags])
            oovs += sum(not w.has_vector for w in X[-1])
            n += len(X[-1])
        print(oovs, n, oovs / n)
        return zip(X, y)

    return _encode(train_sents), _encode(dev_sents), len(tagmap)


def get_word_shape(string):
    shape = spacy.orth.word_shape(string)
    if shape == 'xxxx':
        shape += string[-3:]
    return shape


@plac.annotations(
    nr_sent=("Limit number of training examples", "option", "n", int),
    nr_epoch=("Limit number of training epochs", "option", "i", int),
)
def main(train_loc, dev_loc, checkpoints, nr_epoch=10, nr_sent=0):
    checkpoints = pathlib.Path(checkpoints)
    nlp = spacy.load('en', parser=False, tagger=False, entity=False)
    # Set shape feature
    for word in nlp.vocab:
        word.shape_ = get_word_shape(word.orth_)
    nlp.vocab.lex_attr_getters[SHAPE] = get_word_shape
    train_data, check_data, nr_class = spacy_conll_pos_tags(nlp, train_loc, dev_loc)
    model = EncodeTagger(nr_class, nr_in=nlp.vocab.vectors_length)
    if nr_sent >= 1:
        train_data = train_data[:nr_sent]
    
    with model.begin_training(train_data) as (trainer, optimizer):
        trainer.nb_epoch = nr_epoch
        i = 0
        for examples, truth in trainer.iterate(model, train_data, check_data,
                                               nb_epoch=trainer.nb_epoch):
            truth = model.ops.flatten(truth)
            guess, finish_update = model.begin_update(examples, dropout=trainer.dropout)
            gradient, loss = trainer.get_gradient(guess, truth)
            optimizer.set_loss(loss)
            finish_update(gradient, optimizer)
            i += 1
            if not i % 1000:
                with (checkpoints / ('%d.pickle'%i)).open('wb') as file_:
                    pickle.dump(model, file_, -1)
    print("End", score_model(model, check_data))


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
