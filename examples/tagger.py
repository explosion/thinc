import plac
from os import path
import os
import sys
from collections import defaultdict
import random
import time
import pickle

random.seed(0)


from thinc.nn import NeuralNet


START = ['-START-', '-START2-']
END = ['-END-', '-END2-']


class DefaultList(list):
    """A list that returns a default value if index out of bounds."""
    def __init__(self, default=None):
        self.default = default
        list.__init__(self)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return self.default


def pad_tokens(tokens):
    tokens.insert(0, '<start>')
    tokens.append('ROOT')


class Tagger(object):
    def __init__(self, classes=None, load=False):
        self.strings = {}
        self.tagdict = {}
        if classes:
            self.classes = classes
        else:
            self.classes = {}
        self.model = None

    def start_training(self, sentences):
        suffix_width = 5
        prefix_width = 5
        tags_width = 20
        words_width = 50
        #tables = (suffix_width, prefix_width, tags_width, words_width)
        #slots = (0, 1, 2, 2, 3, 3, 0, 3, 3, 0, 3)
        tables = (words_width,tags_width)
        slots = (0,0,0,0,0,1,1)
        input_length = sum(tables[slot] for slot in slots)
        self._make_tagdict(sentences)
        self.model = NeuralNet(
            (input_length, len(self.classes)),
            embed=(tables, slots),
        rho=1e-7, eta=0.1)
    
    def tag(self, words):
        prev, prev2 = START
        tags = DefaultList('') 
        context = START + [self._normalize(w) for w in words] + END
        inverted_classes = {i: tag for tag, i in self.classes.items()}
        for i, word in enumerate(words):
            features = self._get_features(i, word, context, prev, prev2)
            eg = self.model(features)
            tag = inverted_classes[eg.guess]
            tags.append(tag)
            prev2 = prev
            prev = tag
        return tags
    
    def train(self, sentences, save_loc=None, nr_iter=5):
        '''Train a model from sentences, and save it at save_loc. nr_iter
        controls the number of Perceptron training iterations.'''
        self.start_training(sentences)
        for iter_ in range(nr_iter):
            for words, tags in sentences:
                self.train_one(words, tags)
            random.shuffle(sentences)
        self.end_training(save_loc)
    
    def train_one(self, words, tags):
        prev, prev2 = START
        context = START + [self._normalize(w) for w in words] + END
        Xs = []
        ys = []
        inverted_classes = {i: tag for tag, i in self.classes.items()}
        for i, word in enumerate(words):
            feats = self._get_features(i, word, context, prev, prev2)
            if tags[i] not in ('ROOT', '<start>', None):
                ys.append(self.classes[tags[i]])
                Xs.append(feats)
                eg = self.model.Example(feats, label=self.classes[tags[i]])
                self.model(eg)
                guess = inverted_classes[eg.guess]
                best = inverted_classes[eg.best]
            else:
                guess = tags[i]
            prev2 = prev
            prev = guess
            #print word, guess, tags[i]
            #print feats
        if len(Xs):
            batch = self.model.train(Xs, ys)
            #print Xs
            #print ys
            return batch
        else:
            return model.Batch([], [])

    def save(self):
        # Pickle as a binary file
        pickle.dump((self.model.weights, self.tagdict, self.classes),
                    open(PerceptronTagger.model_loc, 'wb'), -1)

    def load(self, loc):
        w_td_c = pickle.load(open(loc, 'rb'))
        self.model.weights, self.tagdict, self.classes = w_td_c
        self.model.classes = self.classes

    def _normalize(self, word):
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        '''Map tokens into a feature representation, implemented as a
        {(slot, id): float} dict. If the features change, a new model must be
        trained.'''
        def _intify(string):
            if string in self.strings:
                return self.strings[string]
            else:
                i = len(self.strings) + 1
                self.strings[string] = i
                return i

        i += len(START)
        features = {}
        features[(0, _intify(word))] = 1
        features[(1, _intify(context[i-1]))] = 1
        features[(2, _intify(context[i+1]))] = 1
        features[(3, _intify(context[i-2]))] = 1
        features[(4, _intify(context[i+2]))] = 1
        features[(5, _intify(prev))] = 1 # Previous tag
        features[(6, _intify(prev2))] = 1 # Prev prev tag
        #features[(0, _intify(word[-3:]))] = 1 # Suffix of word
        #features[(1, _intify(word[0]))] = 1 # Prefix of word
        #features[(3, _intify(prev2))] = 1 # Prev prev tag
        #features[(4, _intify(word))] = 1 # Word
        #features[(5, _intify(context[i-1]))] = 1 # Previous word
        #features[(6, _intify(context[i-1][-3:]))] = 1 # Suffix of previous word
        #features[(7, _intify(context[i-2]))] = 1 # Prev prev word
        #features[(8, _intify(context[i+1]))] = 1 # Next word
        #features[(9, _intify(context[i+1][-3:]))] = 1 # Suffix of next word
        #features[(10, _intify(context[i+2]))] = 1 # Next next word
        return features

    def _make_tagdict(self, sentences):
        '''Make a tag dictionary for single-tag words.'''
        counts = defaultdict(lambda: defaultdict(int))
        for sent in sentences:
            for word, tag in zip(sent[0], sent[1]):
                counts[word][tag] += 1
                if tag not in self.classes:
                    self.classes[tag] = len(self.classes)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
                self.tagdict[word] = self.classes[tag]


def train(tagger, sentences, nr_iter):
    sentences = list(sentences)
    random.shuffle(sentences)
    partition = int(len(sentences) / 10)
    train_sents = sentences[:partition]
    dev_sents = sentences[partition:]
    tagger.start_training(sentences)
    for itn in range(nr_iter):
        loss = 0
        grad_l1 = 0

        for words, gold_tags, _, _1 in train_sents:
            batch = tagger.train_one(words, gold_tags)
            loss += batch.loss
            grad_l1 += batch.l1_gradient
        corr = 0.0
        total = 1e-6
        for words, gold_tags, _, _1 in dev_sents:
            guesses = tagger.tag(words)
            assert len(gold_tags) == len(guesses)
            for guess, gold in zip(gold_tags, guesses):
                corr += guess == gold
            total += len(guesses)
        print itn, '%.3f' % loss, '%.3f' % (corr / total), '%.3f' % grad_l1
        random.shuffle(train_sents)



def read_conll(loc):
    for sent_str in open(loc).read().strip().split('\n\n'):
        lines = [line.split() for line in sent_str.split('\n')]
        words = DefaultList(''); tags = DefaultList('')
        heads = [None]; labels = [None]
        for i, (word, pos, head, label) in enumerate(lines):
            words.append(intern(word))
            #words.append(intern(normalize(word)))
            tags.append(intern(pos))
            heads.append(int(head) if head != '0' else len(lines) + 1)
            labels.append(label)
        pad_tokens(words)
        pad_tokens(tags)
        yield words, tags, heads, labels


def main(model_dir, train_loc, heldout_gold):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    input_sents = [words for words, tags, labels, heads in read_conll(heldout_gold)]
    tagger = Tagger(load=False)
    sentences = list(read_conll(train_loc))
    train(tagger, sentences, nr_iter=100)


if __name__ == '__main__':
    plac.call(main)
