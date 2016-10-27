"""
Feed-forward neural network with word dropout, following
Iyyer et al. 2015,
Deep Unordered Composition Rivals Syntactic Methods for Text Classification
https://www.cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

from itertools import imap
from collections import defaultdict
from pathlib import Path
import plac
import numpy.random
from math import exp

from thinc.neural.nn import NeuralNet
from thinc.linear.avgtron import AveragedPerceptron
from thinc.extra.eg import Example


def read_data(data_dir):
    for subdir, label in (('pos', 1), ('neg', 0)):
        i = 0
        for filename in (data_dir / subdir).iterdir():
            text = filename.open().read()
            if len(text) >= 10:
                yield text, label
                i += 1


def partition(examples, split_size):
    examples = list(examples)
    numpy.random.shuffle(examples)
    n_docs = len(examples)
    split = int(n_docs * split_size)
    return examples[:split], examples[split:]


def minibatch(data, bs=24):
    for i in range(0, len(data), bs):
        yield data[i:i+bs]


def preprocess(text):
    tokens = []
    for line in text.split('\n'):
        line = line.strip().replace('.', '').replace(',', '')
        line = line.replace(';', '').replace('<br />', ' ')
        line = line.replace(':', '').replace('"', '')
        line = line.replace('(', '').replace(')', '')
        line = line.replace('!', '').replace('*', '')
        line = line.replace(' - ', ' ').replace(' -- ', '')
        line = line.replace('?', '')
        tokens.extend(line.lower().split())
    return tokens


class Extractor(object):
    def __init__(self, dropout=0.3, bigrams=False):
        self.dropout = dropout
        self.bigrams = bigrams
        self.vocab = {}

    def __call__(self, text, dropout=True):
        doc = preprocess(text)
        dropout = self.dropout if dropout is True else 0.0
        bow = defaultdict(float)
        all_words = defaultdict(float)
        prev = None
        inc = (1./(1-dropout))
        for word in doc:
            id_ = self.vocab.setdefault(word, len(self.vocab) + 1)
            if numpy.random.random() >= dropout and word.isalpha():
                bow[id_] += inc
                if self.bigrams and prev is not None:
                    bi = self.vocab.setdefault(prev+'_'+word, len(self.vocab) + 1)
                    bow[bi] += inc
                    all_words[bi] += inc
                prev = word
            else:
                prev = None
            all_words[id_] += inc
        if sum(bow.values()) < 1:
            bow = all_words
        # Normalize for frequency and adjust for dropout
        total = sum(bow.values())
        for word, freq in bow.items():
            bow[word] = float(freq) / total
        return bow


class DenseAveragedNetwork(NeuralNet):
    '''A feed-forward neural network, where:
    
    * Input is an embedding layer averaged from the words in a document
    * Widths of all layers are the same, including the input (embedding) layer
    * ReLu non-linearities are applied to the hidden layers
    * Softmax is applied to the output
    * Weights updated with Adagrad
    * Weights initialized with He initialization
    * Dropout is applied at the token level
    '''
    def __init__(self, n_classes, width, depth, get_bow, rho=1e-5, eta=0.005,
                 eps=1e-6, batch_norm=False, update_step='sgd_cm', noise=0.001):
        unigram_width = width
        bigram_width = 0
        nn_shape = tuple([unigram_width + bigram_width] + [width] * depth + [n_classes])
        NeuralNet.__init__(self, nn_shape, embed=((width,bigram_width), (0,1)),
                           rho=rho, eta=eta, update_step=update_step,
                           batch_norm=batch_norm, noise=noise)
        self.get_bow = get_bow

    def Eg(self, text, label=None):
        bow = self.get_feats(text, dropout=bool(label is not None))
        eg = Example(nr_class=self.nr_class, nr_feat=len(bow), is_sparse=True)
        eg.costs = [i != label for i in range(self.nr_class)]
        eg.features = bow
        return eg

    def get_feats(self, text, dropout=True):
        bow = self.get_bow(text, dropout=dropout)
        output = {}
        for word_id, freq in bow.items():
            output[(0, word_id)] = freq
        return output

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


class NeuralNGram(NeuralNet):
    def __init__(self, n_classes, width, depth, window_size, rho=1e-5, eta=0.005,
                 eps=1e-6, norm_type=None, update_step='sgd_cm', noise=0.001):
        self.vocab = {}
        self.window_size = window_size
        unigram_width = int(width / window_size)
        nn_shape = tuple([width] * depth + [n_classes])
        print(unigram_width, nn_shape)
        NeuralNet.__init__(self, nn_shape, embed=((unigram_width,), (0,0,0,0)),
                           rho=rho, eta=eta, update_step=update_step,
                           norm_type=norm_type, noise=noise)

    def Eg(self, text, label=None):
        doc = preprocess(text)
        doc += ['-EOL-'] * (len(doc) % self.window_size)
        doc = [self.vocab.setdefault(word, len(self.vocab) + 1) for word in doc]
        egs = []
        for i in range(0, len(doc)):
            ngram = doc[i:i+self.window_size]
            feats = [(i, id_, 1) for i, id_ in enumerate(ngram)]
            eg = Example(nr_class=self.nr_class, nr_feat=len(feats), is_sparse=True)
            eg.costs = [i != label for i in range(self.nr_class)]
            eg.features = feats
            egs.append(eg)
        return egs

    def __call__(self, egs):
        total = Example(nr_class=self.nr_class)
        scores = [0 for _ in range(self.nr_class)]
        for eg in egs:
            eg = NeuralNet.__call__(self, eg)
            eg_scores = eg.scores
            for i in range(self.nr_class):
                scores[i] += eg_scores[i]
        total.scores = _softmax(scores)
        total.costs = egs[0].costs
        return total

    def update(self, egs):
        total = self(egs)
        label = total.best
        delta_loss = [0.0 for _ in range(self.nr_class)]
        for i in range(self.nr_class):
            delta_loss[i] = total.scores[i] - (1 if i == label else 0)
        for eg in egs:
            eg.costs = delta_loss
            NeuralNet.update(self, eg)
        return total.loss


def _softmax(nums):
    max_ = max(nums)
    nums = [exp(n-max_) for n in nums]
    Z = sum(nums)
    return [n/Z for n in nums]



class BOWTron(AveragedPerceptron):
    def __init__(self, n_classes, get_bow, *args, **kwargs):
        AveragedPerceptron.__init__(self, tuple())
        self.nr_class = n_classes
        self.get_bow = get_bow

    @property
    def nr_weight(self):
        return 1

    @property
    def weights(self):
        return [self.mem.size]

    def Eg(self, text, label=None):
        bow = self.get_feats(text, dropout=bool(label is not None))
        eg = Example(nr_class=self.nr_class, nr_feat=len(bow))
        eg.costs = [i != label for i in range(self.nr_class)]
        eg.features = bow
        return eg

    def get_feats(self, text, dropout=True):
        bow = self.get_bow(text, dropout=dropout)
        output = {}
        for word_id, freq in bow.items():
            output[(0, word_id)] = freq
        return output

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


@plac.annotations(
    data_dir=("Data directory", "positional", None, Path),
    vectors_loc=("Path to pre-trained vectors", "positional", None, Path),
    n_iter=("Number of iterations (epochs)", "option", "i", int),
    width=("Size of hidden layers", "option", "H", int),
    depth=("Depth", "option", "d", int),
    dropout=("Drop-out rate", "option", "r", float),
    rho=("Regularization penalty", "option", "p", float),
    eta=("Learning rate", "option", "e", float),
    batch_norm=("Use batch normalization", "flag", "B"),
    batch_size=("Batch size", "option", "b", int),
    solver=("Solver", "option", "s", str),
    noise=("Gradient noise", "option", "w", float),
)
def main(data_dir, vectors_loc=None, depth=3, width=300, n_iter=5,
         batch_size=24, dropout=0.5, rho=1e-5, eta=0.005, batch_norm=False,
         solver='sgd_cm', noise=0.0):
    n_classes = 2
    print("Initializing model")
    get_bow = Extractor(dropout=dropout, bigrams=False),
    model = NeuralNGram(n_classes, width, depth, 4,
                        update_step=solver, rho=rho, eta=eta, eps=1e-6,
                        norm_type='layer' if batch_norm else None, noise=noise)
    #model = BOWTron(2, Extractor(dropout, bigrams=True))
    print("Read data")
    train_data, dev_data = partition(read_data(data_dir / 'train'), 0.8)
    print("Begin training")
    prev_best = 0
    best_weights = None
    numpy.random.seed(0)
    prev_score = 0.0
    try:
        for epoch in range(n_iter):
            numpy.random.shuffle(train_data)
            train_loss = 0.0
            for text, label in train_data[:1000]:
                eg = model.Eg(text, label)
                train_loss += model.update(eg)
            nr_correct = sum(model(model.Eg(x)).guess == y for x, y in dev_data)
            print(epoch, train_loss, nr_correct / len(dev_data),
                  sum(model.weights) / model.nr_weight)
            model.eta *= 0.9
    except KeyboardInterrupt:
        print("Stopping")
    print("Evaluating")
    eval_data = list(read_data(data_dir / 'test'))
    n_correct = sum(model(model.Eg(x)).guess == y for x, y in eval_data)
    print(n_correct / len(eval_data))
    print("After averaging")
    model.end_training()
    eval_data = list(read_data(data_dir / 'test'))
    n_correct = sum(model(model.Eg(x)).guess == y for x, y in eval_data)
    print(n_correct / len(eval_data))
 

if __name__ == '__main__':
    plac.call(main)
