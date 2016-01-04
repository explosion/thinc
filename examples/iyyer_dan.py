"""
Feed-forward neural network with word dropout, following
Iyyer et al. 2015,
Deep Unordered Composition Rivals Syntactic Methods for Text Classification
https://www.cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

from collections import defaultdict
from pathlib import Path
import plac
import numpy.random

from thinc.nn import NeuralNet
from thinc.eg import Batch


def read_data(data_dir):
    for subdir, label in (('pos', (0, 1)), ('neg', (1, 0))):
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
    def __init__(self, nr_embed, dropout=0.3):
        self.nr_embed = nr_embed
        self.dropout = dropout
        self.vocab = {}

    def __call__(self, text, dropout=None):
        doc = preprocess(text)
        if dropout is None:
            dropout = self.dropout
        bow = defaultdict(float)
        all_words = defaultdict(float)
        for word in doc:
            id_ = self.vocab.setdefault(word, len(self.vocab) + 1)
            if numpy.random.random() >= dropout and word.isalpha():
                bow[id_] += 1
            all_words[id_] += 1
        if sum(bow.values()) < 1:
            bow = all_words
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
                 eps=1e-6, bias=0.0):
        nn_shape = tuple([width] + [width] * depth + [n_classes])
        NeuralNet.__init__(self, nn_shape, embed=((width,), (0,)),
                           rho=rho, eta=eta, eps=eps, bias=bias)
        self.get_bow = get_bow

    def train(self, batch):
        loss = 0.0
        X = [self.get_bow(text) for text, _ in batch]
        y = [label for _, label in batch]
        batch = NeuralNet.train(self, X, y)
        return batch

    def predict(self, text):
        word_ids = self.get_bow(text, dropout=0.0)
        eg = self.Example(word_ids)
        self(eg)
        return eg

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
    bias=("Initialize biases to", "option", "B", float),
    batch_size=("Batch size", "option", "b", int),
)
def main(data_dir, vectors_loc=None, depth=2, width=300, n_iter=5,
         batch_size=24, dropout=0.5, rho=1e-5, eta=0.005, bias=0.0):
    n_classes = 2
    print("Initializing model")
    model = DenseAveragedNetwork(n_classes, width, depth, Extractor(width, dropout),
                                 rho=rho, eta=eta, eps=1e-6, bias=bias)
    print(model.widths)
    print(model.nr_weight)
    print("Read data")
    train_data, dev_data = partition(read_data(data_dir / 'train'), 0.8)
    print("Begin training")
    prev_best = 0
    best_weights = None
    numpy.random.seed(0)
    for epoch in range(n_iter):
        numpy.random.shuffle(train_data)
        train_loss = 0.0
        avg_grad = 0.0
        for X_y in minibatch(train_data, bs=batch_size):
            batch = model.train(X_y)
            if str(batch.loss) == 'nan':
                raise Exception(batch.gradient)
            train_loss += batch.loss
            avg_grad += batch.l1_gradient
            #avg_grad += sum(abs(g) for g in batch.gradient) / model.model.nr_weight
        n_correct = sum(y[model.predict(x).guess] == 0 for x, y in dev_data)
        print(epoch, train_loss, n_correct / len(dev_data),
              sum(model.weights) / model.nr_weight,
              avg_grad)
        if n_correct >= prev_best:
            prev_best = n_correct
    print("Evaluating")
    eval_data = list(read_data(data_dir / 'test'))
    n_correct = sum(y[model.predict(x).guess] == 0 for x, y in eval_data)
    print(n_correct / len(eval_data))
 

if __name__ == '__main__':
    #import cProfile
    #import pstats
    #cProfile.runctx("main(Path('/Users/matt/repos/sentiment_tutorial/data/aclImdb'))", globals(), locals(), "Profile.prof")
    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats(100)

    plac.call(main)
