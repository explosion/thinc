import random
import io
from collections import Counter


ANCORA_TRAIN_LOC = 'data/es_ancora-ud-train.conllu.txt'
ANCORA_DEV_LOC = 'data/es_ancora-ud-dev.conllu.txt'


def conll_pos_tags(train_loc=ANCORA_TRAIN_LOC, dev_loc=ANCORA_DEV_LOC):
    train_sents = list(read_conll(train_loc))
    dev_sents = list(read_conll(dev_loc))
    tagmap = {}
    freqs = Counter()
    for words, tags, heads, labels in train_sents:
        for tag in tags:
            tagmap.setdefault(tag, len(tagmap))
        for word in words:
            freqs[word] += 1
    vocab = {word: i for i, (word, freq) in enumerate(freqs.most_common())
             if (freq >= 10)}

    def _encode(sents):
        X = []
        y = []
        for words, tags, heads, labels in sents:
            X.append([vocab.get(word, len(vocab)) for word in words])
            y.append([tagmap[tag] for tag in tags])
        return zip(X, y)

    return _encode(train_sents), _encode(dev_sents), len(tagmap)


def read_conll(loc):
    n = 0
    with io.open(loc, encoding='utf8') as file_:
        sent_strs = file_.read().strip().split('\n\n')
    for sent_str in sent_strs:
        lines = [line.split() for line in sent_str.split('\n')
                 if not line.startswith('#')]
        words = []
        tags = []
        heads = [None]
        labels = [None]
        for i, pieces in enumerate(lines):
            if len(pieces) == 4:
                word, pos, head, label = pieces
            else:
                idx, word, lemma, pos1, pos, morph, head, label, _, _2 = pieces
            words.append(word)
            tags.append(pos)
            heads.append(int(head) if head != '0' else len(lines) + 1)
            labels.append(label)
        yield words, tags, heads, labels


def keras_mnist():
    from keras.datasets import mnist
    from keras.utils import np_utils

    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    train_data = zip(X_train, y_train)
    nr_train = len(train_data)
    random.shuffle(train_data)
    heldout_data = train_data[:int(nr_train * 0.1)] 
    train_data = train_data[len(heldout_data):]
    test_data = zip(X_test, y_test)
    return train_data, heldout_data, test_data
