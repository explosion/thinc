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

from thinc.neural.nn import NeuralNet


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
    def __init__(self, nr_embed, dropout=0.3):
        self.nr_embed = nr_embed
        self.dropout = dropout
        self.vocab = {}

    def __call__(self, text, dropout=True, bigrams=False):
        doc = preprocess(text)
        dropout = self.dropout if dropout is True else 0.0
        bow = defaultdict(float)
        all_words = defaultdict(float)
        prev = None
        for word in doc:
            id_ = self.vocab.setdefault(word, len(self.vocab) + 1)
            if numpy.random.random() >= dropout and word.isalpha():
                bow[id_] += 1
                if bigrams and prev is not None:
                    bi = self.vocab.setdefault(prev+'_'+word, len(self.vocab) + 1)
                    bow[bi] += 1
                    all_words[bi] += 1
                prev = word
            else:
                prev = None
            all_words[id_] += 1
        if sum(bow.values()) < 1:
            bow = all_words
        # Normalize for frequency and adjust for dropout
        total = sum(bow.values())
        total *= 1-dropout
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
                 eps=1e-6, update_step='adadelta'):
        nn_shape = tuple([width] + [width] * depth + [n_classes])
        NeuralNet.__init__(self, nn_shape, embed=((width,), (0,)),
                           rho=rho, eta=eta, eps=eps,
                           update_step=update_step)
        self.get_bow = get_bow

    def train(self, text, label):
        self.eg.reset()
        self.eg.features = self.get_feats(text)
        self.eg.costs = [i != label for i in range(self.eg.nr_class)]
        eg = self.train_example(self.eg)
        return eg

    def predict(self, text):
        self.eg.reset()
        self.eg.features = self.get_feats(text, dropout=False)
        return self.predict_example(self.eg)

    def get_feats(self, text, dropout=True):
        word_ids = self.get_bow(text, dropout=dropout)
        return {(0, word_id): freq for (word_id, freq) in word_ids.items()}

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
    solver=("Solver", "option", "s", str),
)
def main(data_dir, vectors_loc=None, depth=2, width=300, n_iter=5,
         batch_size=24, dropout=0.5, rho=1e-5, eta=0.005, bias=0.0, solver='sgd'):
    n_classes = 2
    print("Initializing model")
    model = DenseAveragedNetwork(n_classes, width, depth, Extractor(width, dropout),
                                 update_step=solver, rho=rho, eta=eta, eps=1e-6)
    print(model.widths)
    print(model.nr_weight)
    print("Read data")
    train_data, dev_data = partition(read_data(data_dir / 'train'), 0.8)
    print("Begin training")
    prev_best = 0
    best_weights = None
    numpy.random.seed(0)
    for i, (w, b) in enumerate(model.layers):
        print("Layer %d means:" % i, sum(w)/len(w), sum(b)/len(b))

    for epoch in range(n_iter):
        numpy.random.shuffle(train_data)
        train_loss = 0.0
        avg_grad = 0.0
        for text, label in train_data:
            eg = model.train(text, label)
            #print(list(model.layers[-1])[1])
            train_loss += eg.loss
            avg_grad += model.l1_gradient
        n_correct = sum(model.predict(x).guess == y for x, y in dev_data)
        print(epoch, train_loss, n_correct / len(dev_data),
              sum(model.weights) / model.nr_weight,
              avg_grad)
        if n_correct >= prev_best:
            prev_best = n_correct
    print("Evaluating")
    eval_data = list(read_data(data_dir / 'test'))
    n_correct = sum(model.predict(x).guess == y for x, y in eval_data)
    print(n_correct / len(eval_data))
 

if __name__ == '__main__':
    #import cProfile
    #import pstats
    #cProfile.runctx("main(Path('/Users/matt/repos/sentiment_tutorial/data/aclImdb'))", globals(), locals(), "Profile.prof")
    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats(100)

    plac.call(main)
