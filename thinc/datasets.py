import random


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
