import plac
import pickle

from thinc.neural import Model, ReLu, Softmax, Embed
from thinc.neural._classes.rnn import BiLSTM
from thinc.api import chain, concatenate, layerize, with_flatten
from thinc.neural.util import to_categorical


class Example(object):
    def __init__(self, tokens, labels, n_labels):
        self.tokens = tokens
        self.labels = labels
        self.n_labels = n_labels


def build_model(n_tags, n_words, word_width, tag_width, hidden_width):
    with Model.define_operators({'|': concatenate, '>>': chain}):
        words_model = (
            with_flatten(
                Embed(word_width, word_width, n_words), pad=0
            )
            >> BiLSTM(word_width, word_width)
            >> flatten_add_lengths
            >> getitem(0)
            >> Affine(hidden_width, word_width * 2)
            >> pad_and_reshape
        )

        tags_model = (
            Embed(tag_width, tag_width, n_tags)
            >> Affine(hidden_width, tag_width)
        )

        state_model = Affine(hidden_width, hidden_width)

        output_model = Softmax(n_tags, hidden_width)
        words_model.nO = hidden_width
        state_model.nO = hidden_width
        output_model.nO = n_tags

    def fwd_step(features, drop=0.):
        word_feats, prev_tags, prev_state = features
        tag_feats, bp_tags = tags_model.begin_update(prev_tags, drop=drop)
        state_feats, bp_state = state_model.begin_update(prev_state, drop=drop)

        preact = word_feats + tag_feats + state_feats
        nonlin = preact > 0
        state = preact * nonlin
        scores, bp_scores = output_model.begin_update(state, drop=drop)

        def bwd_step(d_scores, d_next_state, sgd=None):
            d_state = d_next_state + bp_scores(d_scores, sgd=sgd)
            d_state *= nonlin
            bp_tags(d_state, sgd=sgd)
            d_prev_state = bp_state(d_state, sgd=sgd)
            return d_state, d_prev_state
        (state, scores), bwd_step
    return words_model, fwd_step


def fit_batch(model, sgd, word_ids, true_tags):
    words_model, tagger = model
    ops = words_model.ops

    words, bp_words = words_model.begin_update(word_ids)

    nS = words.shape[0]
    nB = words.shape[1]
    nH = words.shape[2] # Hidden dimension
    nC = tagger.nO  # Number of tag classes
    prev_tags = ops.allocate((nB,), dtype='i')
    prev_state = ops.allocate((nB, nH))
    d_next_state = ops.allocate((nB, nH))

    for i in range(nS):
        features = (words[i], prev_tags, prev_state)
        (state, scores), bp_step = tagger.begin_update(features, drop=drop)

        d_scores = (scores - truths[i]) / truths.shape[0]
        loss += d_scores.sum() / (nB * nS * nC)

        d_words[i], d_next_state = bp_step((d_scores, d_next_state), sgd=sgd)
        prev_tags = scores.argmax(axis=1)
        prev_state = state
    bp_words(d_words, sgd=sgd)
    return loss


def pad_batch(ops, seqs):
    nB = len(seqs)
    nS = max([len(seq) for seq in seqs])
    arr = ops.allocate((nB, nS), dtype='i')
    for i, seq in enumerate(seqs):
        arr[i, :len(seq)] = ops.asarray(seq)
    return arr
 

def train(model, train_words, train_tags, n_epoch=5):
    with model[0].begin_training(train_words, train_tags) as (trainer, optimizer):
        losses = [0.]
        def print_loss():
            print(len(losses), losses[-1])
        trainer.nb_epoch = n_epoch
        trainer.batch_size = 4
        trainer.each_epoch.append(print_loss)
        for words_batch, tags_batch in trainer.iterate(train_words, train_tags):
            tags_batch = pad_batch(model[0].ops, tags_batch)
            losses[-1] += fit_batch(model, optimizer, 
                            words_batch, tags_batch)


def main(data_loc='wsj.pkl', word_width=50, tag_width=51, hidden_width=52):
    data, n_types, n_labels = pickle.load(open('wsj.pkl', 'rb'))
    model = build_model(n_labels, n_types, word_width, tag_width, hidden_width)
    train_words = [model[0].ops.asarray(seq.tokens) for seq in data]
    train_tags = [model[0].ops.asarray(seq.labels) for seq in data]
    train(model, train_words, train_tags, n_epoch=5)


if __name__ == '__main__':
    plac.call(main)
