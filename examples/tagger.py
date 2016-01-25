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


START = ['-START-'] * 5
END = ['-END-'] * 5


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


class FeatureExtractor(object):
    def __init__(self, char_width=5, word_width=10, tag_width=5, chars_per_word=10,
            word_context=(-2, -1, 0, 1, 2), tag_context=(-2, -1)):
        self.char_width = char_width
        self.tag_width = tag_width
        self.word_width = word_width
        self.chars_per_word = chars_per_word
        self.word_context = word_context
        self.tag_context = tag_context
        self.strings = {}
        self.tables = (self.char_width, self.word_width, self.tag_width)
        char_slots = [0] * self.chars_per_word
        word_slots = [1 for _ in self.word_context]
        tag_slots = [2 for _ in self.tag_context]
        self.slots = char_slots + word_slots + tag_slots
        print(self.tables)
        print(self.slots)

    @property
    def input_length(self):
        return sum(self.tables[slot] for slot in self.slots)

    def __call__(self, i, word, context, prev_tags):
        i += len(START)
        features = []
        if self.chars_per_word > 0:
            if len(word) > self.chars_per_word:
                split = self.chars_per_word / 2
                word = word[:split] + word[-split:]
            else:
                word = word.ljust(self.chars_per_word, ' ')
            # Character features
            assert len(word) == self.chars_per_word, repr(word)
            features = [(len(features), ord(c), 1.0) for c in word]
        else:
            features = []
        for position in self.word_context:
            features.append((len(features), self._intify(context[i+position]), 1.0))
        for position in self.tag_context:
            features.append((len(features), self._intify(prev_tags[position]), 1.0))
        return features
    
    def _intify(self, string):
        if string in self.strings:
            return self.strings[string]
        else:
            i = len(self.strings) + 1
            self.strings[string] = i
            return i


class CharacterExtracter(object):
    def __init__(self, chars_per_word=10, char_width=5, nr_left=1, nr_right=1):
        self.char_width = char_width
        self.chars_per_word = chars_per_word
        self.nr_right = nr_right
        self.nr_left = nr_left
        self.word_context = [-(i+1) for i in range(nr_left)] + [0] + [i+1 for i in range(nr_right)]
        self.strings = {}
        self.tables = (self.char_width,)
        self.slots = []
        for _ in self.word_context:
            self.slots.extend([0] * self.chars_per_word)
 
    @property
    def input_length(self):
        return self.char_width * self.chars_per_word * (1 + self.nr_left + self.nr_right)

    def __call__(self, i, word, context, prev_tags):
        i += len(START)
        features = []
        for position in self.word_context:
            word = context[i+position]
            if len(word) > self.chars_per_word:
                split = self.chars_per_word / 2
                word = word[:split+1] + word[-split:]
            else:
                word = word.ljust(self.chars_per_word, ' ')
            for c in word:
                features.append((len(features), ord(c), 1.0 if c != ' ' else 0))
        return features
 

class Tagger(object):
    def __init__(self, depth, hidden_width, extractor, learn_rate=0.01, L2=1e6, solver='adam',
        classes=None, load=False):
        self.ex = extractor
        self.tagdict = {}
        if classes:
            self.classes = classes
        else:
            self.classes = {}
        self.depth = depth
        self.hidden_width = hidden_width
        self.learn_rate = learn_rate
        self.L2 = L2
        self.solver = solver
        self.model = None

    def start_training(self, sentences):
        self._make_tagdict(sentences)
        input_length = self.ex.input_length
        widths = [input_length] + [self.hidden_width] * self.depth + [len(self.classes)]
        self.model = NeuralNet(
            widths,
            embed=(self.ex.tables, self.ex.slots),
            rho=self.L2, eta=self.learn_rate, update_step=self.solver)
        print(self.model.widths)
    
    def tag(self, words):
        tags = DefaultList('') 
        context = START + [w for w in words] + END
        inverted_classes = {i: tag for tag, i in self.classes.items()}
        eg = self.model.Example([])
        for i, word in enumerate(words):
            eg.wipe(self.model.widths)
            features = self.ex(i, word, context, tags)
            eg.set_features(features)
            eg = self.model.predict_example(eg)
            tag = inverted_classes[eg.guess]
            tags.append(tag)
        return tags
    
    def train_one(self, words, tags):
        tag_history = DefaultList('') 
        #context = START + [self._normalize(w) for w in words] + END
        context = START + [w for w in words] + END
        Xs = []
        ys = []
        inverted_classes = {i: tag for tag, i in self.classes.items()}
        loss = 0.0
        eg = self.model.Example([])
        for i, word in enumerate(words):
            eg.wipe(self.model.widths)
            features = self.ex(i, word, context, tag_history)
            eg.set_features(features)
            eg.set_label(self.classes[tags[i]])
            eg = self.model.train_example(eg)
            tag_history.append(inverted_classes[eg.guess])
            loss += eg.loss
        return loss

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


def train(tagger, sentences, nr_iter=100):
    sentences = list(sentences)
    random.shuffle(sentences)
    partition = int(len(sentences) / 10)
    train_sents = sentences[partition:]
    dev_sents = sentences[:partition]
    tagger.start_training(sentences)
    for itn in range(nr_iter):
        loss = 0
        nr_words = 0
        for words, gold_tags, _, _1 in train_sents:
            loss += tagger.train_one(words, gold_tags)
            nr_words += len(words)
        acc = evaluate(tagger, dev_sents)
        print itn, '%.3f' % (loss / nr_words), '%.3f' % acc
        random.shuffle(train_sents)


def evaluate(tagger, sentences):
    corr = 0.0
    total = 1e-6
    for words, gold_tags, _, _1 in sentences:
        guesses = tagger.tag(words)
        assert len(gold_tags) == len(guesses)
        for guess, gold in zip(gold_tags, guesses):
            corr += guess == gold
            total += 1
    return corr / total
 

def read_conll(loc):
    n = 0
    for sent_str in open(loc).read().strip().split('\n\n'):
        lines = [line.split() for line in sent_str.split('\n')]
        words = DefaultList(''); tags = DefaultList('')
        heads = [None]; labels = [None]
        for i, pieces in enumerate(lines):
            if len(pieces) == 4:
                word, pos, head, label = pieces
            else:
                idx, word, lemma, pos1, pos, morph, head, label, _, _2 = pieces
            words.append(intern(word))
            tags.append(intern(pos))
            heads.append(int(head) if head != '0' else len(lines) + 1)
            labels.append(label)
        yield words, tags, heads, labels


@plac.annotations(
    nr_iter=("Number of iterations", "option", "i", int),
    depth=("Number of hidden layers", "option", "d", int),
    hidden_width=("Number of dimensions for hidden layers", "option", "D", int),
    learn_rate=("Number of hidden layers", "option", "e", float),
    L2=("L2 regularization penalty", "option", "r", float),
    solver=("Optimization algorithm","option", "s", str),
    word_width=("Number of dimensions for word embeddings", "option", "w", int),
    char_width=("Number of dimensions for char embeddings", "option", "c", int),
    tag_width=("Number of dimensions for tag embeddings", "option", "t", int),
    chars_per_word=("Number of characters to give the word", "option", "C", int),
    left_words=("Number of words from the preceding context", "option", "L", int),
    right_words=("Number of words from the following context", "option", "R", int),
    left_tags=("Number of tags from the preceding context", "option", "T", int),
)
def main(model_dir, train_loc, dev_loc,
         depth=2, L2=1e-6, learn_rate=0.01, solver="adam",
         hidden_width=100,
         word_width=10, char_width=5, tag_width=5,
         chars_per_word=0,
         left_words=2, right_words=2, left_tags=2,
         nr_iter=1):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    word_context = [-(i+1) for i in range(left_words)] + [0] + [i+1 for i in range(right_words)]
    tag_context = [-i for i in range(left_tags)]
    print("Word context", word_context)
    ex = FeatureExtractor(char_width, word_width, tag_width,
                          chars_per_word,
                          word_context, tag_context)
    #ex = CharacterExtracter(char_width=char_width, chars_per_word=chars_per_word,
    #                        nr_left=left_words, nr_right=right_words)
    tagger = Tagger(depth, hidden_width, ex,
                learn_rate=learn_rate,
                solver=solver,
                L2=L2,
                load=False)
    sentences = list(read_conll(train_loc))
    train(tagger, sentences, nr_iter=nr_iter)
    print evaluate(tagger, list(read_conll(dev_loc)))


if __name__ == '__main__':
    #import cProfile
    #import pstats
    #cProfile.runctx("main('/Users/matt/repos/thinc/parsers/', '/Users/matt/work_data/ym03_deps/train.tab', '/Users/matt/work_data/ym03_deps/dev.tab')", globals(), locals(), "Profile.prof")
    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats(100)
    plac.call(main)
