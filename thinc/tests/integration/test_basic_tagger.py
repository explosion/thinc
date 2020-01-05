import pytest
import random
from murmurhash import hash_unicode
from thinc.util import to_categorical
from thinc.model import Model
from thinc.layers.relu import ReLu
from thinc.layers.softmax import Softmax
from thinc.layers.hashembed import HashEmbed
from thinc.layers.extractwindow import ExtractWindow
from thinc.layers.chain import chain
from thinc.layers.with_list2array import with_list2array
from thinc.optimizers import Adam
import ml_datasets


@pytest.fixture(scope="module")
def ancora():
    train_data, check_data, nr_class = ml_datasets.ud_ancora_pos_tags()
    train_X, train_y = zip(*train_data)
    dev_X, dev_y = zip(*check_data)
    nb_tag = max(max(y) for y in train_y) + 1
    train_y = [to_categorical(y, nb_tag) for y in train_y]
    dev_y = [to_categorical(y, nb_tag) for y in dev_y]
    return (train_X[:100], train_y[:100]), (dev_X, dev_y)


@pytest.fixture(scope="module")
def train_X(ancora):
    (train_X, train_y), (dev_X, dev_y) = ancora
    return train_X


@pytest.fixture(scope="module")
def train_Y(ancora):
    (train_X, train_y), (dev_X, dev_y) = ancora
    return train_y


@pytest.fixture(scope="module")
def dev_X(ancora):
    (train_X, train_y), (dev_X, dev_y) = ancora
    return dev_X


@pytest.fixture(scope="module")
def dev_Y(ancora):
    (train_X, train_y), (dev_X, dev_y) = ancora
    return dev_y


def create_embed_relu_relu_softmax(depth, width, vector_length):
    with Model.define_operators({">>": chain}):
        model = strings2arrays() >> with_list2array(
            HashEmbed(width, vector_length)
            >> ExtractWindow(window_size=1)
            >> ReLu(width, width * 3)
            >> ReLu(width, width)
            >> Softmax(17, width)
        )
    return model


def strings2arrays():
    def strings2arrays_forward(model, Xs, is_train):
        hashes = [[hash_unicode(word) for word in X] for X in Xs]
        arrays = [model.ops.asarray(h, dtype="uint64") for h in hashes]
        arrays = [array.reshape((-1, 1)) for array in arrays]
        return arrays, lambda dX: dX

    return Model("strings2arrays", strings2arrays_forward)


@pytest.fixture(params=[create_embed_relu_relu_softmax])
def create_model(request):
    return request.param


def evaluate_tagger(model, dev_X, dev_Y, batch_size):
    correct = 0.0
    total = 0.0
    for i in range(0, len(dev_X), batch_size):
        Yh = model.predict(dev_X[i : i + batch_size])
        Y = dev_Y[i : i + batch_size]
        for j in range(len(Yh)):
            correct += (Yh[j].argmax(axis=1) == Y[j].argmax(axis=1)).sum()
            total += Yh[j].shape[0]
    return correct / total


def get_shuffled_batches(Xs, Ys, batch_size):
    zipped = list(zip(Xs, Ys))
    random.shuffle(zipped)
    for i in range(0, len(zipped), batch_size):
        batch_X, batch_Y = zip(*zipped[i : i + batch_size])
        yield list(batch_X), list(batch_Y)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("depth", "width", "vector_width", "nb_epoch"), [(2, 32, 16, 5)]
)
def test_small_end_to_end(
    depth, width, vector_width, nb_epoch, create_model, train_X, train_Y, dev_X, dev_Y
):
    batch_size = 8
    model = create_model(depth, width, vector_width).initialize()
    optimizer = Adam(0.001)
    losses = []
    scores = []
    for i in range(nb_epoch):
        losses.append(0.0)
        for X, Y in get_shuffled_batches(train_X, train_Y, batch_size):
            Yh, backprop = model.begin_update(X)
            d_loss = []
            for i in range(len(Yh)):
                d_loss.append(Yh[i] - Y[i])
                losses[-1] += ((Yh[i] - Y[i]) ** 2).sum()
            backprop(d_loss)
            model.finish_update(optimizer)
        scores.append(evaluate_tagger(model, dev_X, dev_Y, batch_size))
    assert losses[-1] < losses[0]
    assert scores[-1] > scores[0]
