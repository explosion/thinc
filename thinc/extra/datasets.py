import random
import io
from collections import Counter
import os.path

from ._vendorized.keras_data_utils import get_file


GITHUB = 'https://github.com/UniversalDependencies/'
ANCORA_1_4_ZIP = '{github}/{ancora}/archive/r1.4.zip'.format(
    github=GITHUB, ancora='UD_Spanish-AnCora')
EWTB_1_4_ZIP = '{github}/{ewtb}/archive/r1.4.zip'.format(
    github=GITHUB, ewtb='UD_English')


def ancora_pos_tags():
    data_dir = get_file('UD_Spanish-AnCora-r1.4', ANCORA_1_4_ZIP,
                        unzip=True)
    train_loc = os.path.join(data_dir, 'es_ancora-ud-train.conllu')
    dev_loc = os.path.join(data_dir, 'es_ancora-ud-dev.conllu')
    return ud_pos_tags(train_loc, dev_loc)


def ewtb_pos_tags():
    data_dir = get_file('UD_English-r1.4', EWTB_1_4_ZIP, unzip=True)
    train_loc = os.path.join(data_dir, 'en-ud-train.conllu')
    dev_loc = os.path.join(data_dir, 'en-ud-dev.conllu')
    return ud_pos_tags(train_loc, dev_loc, encode_tags=False, encode_words=False)


def ud_pos_tags(train_loc, dev_loc, encode_tags=True, encode_words=True):
    train_sents = list(read_conll(train_loc))
    dev_sents = list(read_conll(dev_loc))
    tagmap = {}
    freqs = Counter()
    for words, tags in train_sents:
        for tag in tags:
            tagmap.setdefault(tag, len(tagmap))
        for word in words:
            freqs[word] += 1
    vocab = {word: i for i, (word, freq) in enumerate(freqs.most_common())
             if (freq >= 10)}

    def _encode(sents):
        X = []
        y = []
        for words, tags  in sents:
            if encode_words:
                X.append([vocab.get(word, len(vocab)) for word in words])
            else:
                X.append(words)
            if encode_tags:
                y.append([tagmap[tag] for tag in tags])
            else:
                y.append(tags)
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
        for i, pieces in enumerate(lines):
            if len(pieces) == 4:
                word, pos, head, label = pieces
            else:
                idx, word, lemma, pos1, pos, morph, head, label, _, _2 = pieces
            words.append(word)
            tags.append(pos)
        yield words, tags


def mnist():
    from ._vendorized.keras_datasets import load_mnist

    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = load_mnist()

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
