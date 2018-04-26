import plac
import pickle
import contextlib

from thinc.v2v import Model, ReLu, Softmax, Affine
from thinc.i2v import Embed
from thinc.t2t import BiLSTM
from thinc.api import chain, concatenate, layerize, with_flatten
from thinc.api import flatten_add_lengths, wrap
from thinc.neural.util import to_categorical

def getitem(i):
    def getitem_fwd(Xs, drop=0.):
        length = len(Xs)
        def getitem_bwd(dX, sgd=None):
            output = [None] * length
            output[i] = dX
            return output
        return Xs[i], getitem_bwd
    return layerize(getitem_fwd)


class Example(object):
    def __init__(self, tokens, labels, n_labels):
        self.tokens = tokens
        self.labels = labels
        self.n_labels = n_labels


def init_models(n_tags, n_words, widths):
    word_width, tag_width, hidden_width = widths
    with Model.define_operators({'|': concatenate, '>>': chain}):
        word_model = (
            with_flatten(
                Embed(word_width, word_width, n_words), pad=0
            )
            >> BiLSTM(word_width, residual=True)
            >> with_flatten(
                Affine(hidden_width, word_width*2))
        )
        
        state_model = Affine(hidden_width, hidden_width)

        tags_model = (
            Embed(hidden_width, tag_width, n_tags)
        )

        output_model = Softmax(n_tags, hidden_width)
    return word_model, TaggerModel(tags_model, state_model, output_model)


def TaggerModel(tags_model, state_model, output_model):
    def tagger_fwd(words_prevtags_prevstate, drop=0.):
        word_feats, prev_tags, prev_state = words_prevtags_prevstate
        tag_feats, bp_tags = tags_model.begin_update(prev_tags, drop=drop)
        state_feats, bp_state = state_model.begin_update(prev_state, drop=drop)

        #preact = word_feats + state_feats + tag_feats
        #nonlin = preact > 0
        #state = preact * nonlin
        #state = preact
        state = word_feats
        scores, bp_scores = output_model.begin_update(state, drop=drop)

        def tagger_bwd(d_scores_d_next_state, sgd=None):
            d_scores, d_next_state = d_scores_d_next_state
            #d_state = d_next_state + bp_scores(d_scores, sgd=sgd)
            d_state = bp_scores(d_scores, sgd=sgd)
            #d_state *= nonlin
            bp_tags(d_state, sgd=sgd)
            d_prev_state = bp_state(d_state, sgd=sgd)
            return d_prev_state, d_state
        return (scores, state), tagger_bwd
    model = wrap(tagger_fwd, tags_model, state_model, output_model)
    model.nO = output_model.nO
    return model


class Tagger(object):
    def __init__(self, n_tags, n_types, word_width, tag_width, hidden_width):
        widths = (word_width, tag_width, hidden_width)
        self.embed, self.tag = init_models(n_tags, n_types, widths)
        self.nr_tag = n_tags
        self.n_types = n_types
        self.word_width = word_width
        self.tag_width = tag_width
        self.hidden_width = hidden_width

    @property
    def ops(self):
        return self.embed.ops

    @contextlib.contextmanager
    def begin_training(self, train_words, train_tags):
        train_words = train_words[:5]
        train_tags = pad_batch(self.ops, train_tags[:5])
        with self.embed.begin_training(train_words) as (trainer, optimizer):
            optimizer.b2 = 0.99
            optimizer.alpha = 0.001
            yield trainer, optimizer

    def update(self, word_ids, truths, drop=0., sgd=None):
        lengths = [len(w) for w in word_ids]
        truths = pad_batch(self.ops, truths)
        word_embeds, bp_words = self.embed.begin_update(word_ids, drop=drop)
        word_embeds = pad_batch(self.ops, word_embeds, shape=word_embeds[0].shape)
        d_word_embeds = self.ops.allocate(word_embeds.shape)

        nr_state = word_embeds.shape[0]
        nr_batch = word_embeds.shape[1]
        nr_hidden = self.hidden_width

        prev_tags = self.ops.allocate((nr_batch,), dtype='i')
        prev_state = self.ops.allocate((nr_batch, nr_hidden))
        d_next_state = self.ops.allocate((nr_batch, nr_hidden))

        loss = 0.
        acc = 0.
        for i in range(nr_state):
            features = (word_embeds[i], prev_tags, prev_state)
            (scores, state), bp_step = self.tag.begin_update(features, drop=drop)

            d_scores = (scores - to_categorical(truths[i], self.nr_tag)) / truths.shape[0]
            loss += d_scores.sum()
            d_word_embeds[i], d_next_state = bp_step((d_scores, d_next_state), sgd=sgd)
            prev_tags = scores.argmax(axis=1)
            acc += (prev_tags == truths[i]).sum()
            prev_state = state
        d_word_embeds = unpad_batch(self.ops, d_word_embeds, lengths)
        bp_words(d_word_embeds, sgd=sgd)
        return loss, acc


def pad_batch(ops, seqs, shape=None):
    nB = len(seqs)
    nS = max([len(seq) for seq in seqs])
    if shape is None:
        arr = ops.allocate((nB, nS), dtype='i')
    else:
        arr = ops.allocate((nB, nS) + shape[1:])
    for i, seq in enumerate(seqs):
        arr[i, :len(seq)] = ops.asarray(seq)
    return arr


def unpad_batch(ops, arr, lengths):
    seqs = []
    for i in range(arr.shape[0]):
        seqs.append(arr[i, :lengths[i]])
    return seqs


def main(data_loc='wsj.pkl', word_width=50, tag_width=50, hidden_width=51,
         n_epoch=10, drop=0.0):
    data, n_types, n_labels = pickle.load(open('wsj.pkl', 'rb'))
    tagger = Tagger(n_labels, n_types, word_width, tag_width, hidden_width)
    train_words = [tagger.ops.asarray(seq.tokens) for seq in data]
    train_tags = [tagger.ops.asarray(seq.labels) for seq in data]
    nr_word = sum(len(w) for w in train_words)
    print(nr_word)
    with tagger.begin_training(train_words, train_tags) as (trainer, optimizer):
        losses = [0.]
        accs = [0.]
        def print_loss():
            print(len(losses), losses[-1], accs[-1] / nr_word)
            losses.append(0.)
            accs.append(0.)
        trainer.nb_epoch = n_epoch
        trainer.batch_size = 5
        trainer.each_epoch.append(print_loss)
        for words_batch, tags_batch in trainer.iterate(train_words, train_tags):
            loss_acc = tagger.update(words_batch, tags_batch,
                            drop=drop, sgd=optimizer)
            losses[-1] += loss_acc[0]
            accs[-1] += loss_acc[1]


if __name__ == '__main__':
    plac.call(main)
