# https://raw.githubusercontent.com/fchollet/keras/master/keras/datasets/mnist.py
# Copyright Francois Chollet, Google, others (2015)
# Under MIT license

import gzip
from .keras_data_utils import get_file
import sys


def load_mnist(path='mnist.pkl.gz'):
    from six.moves import cPickle
    path = get_file(path,
        origin='https://s3.amazonaws.com/img-datasets/mnist.pkl.gz')

    if path.endswith('.gz'):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='bytes')

    f.close()
    return data  # (X_train, y_train), (X_test, y_test)
