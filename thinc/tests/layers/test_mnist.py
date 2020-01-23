import pytest
from thinc.api import Model, ReLu, Softmax, chain, clone, Adam
import ml_datasets


@pytest.fixture(scope="module")
def mnist(limit=1000):
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
    return (train_X[:limit], train_Y[:limit]), (dev_X[:limit], dev_Y[:limit])


@pytest.fixture(scope="module")
def train_X(mnist):
    (train_X, train_y), (dev_X, dev_y) = mnist
    return train_X


@pytest.fixture(scope="module")
def train_Y(mnist):
    (train_X, train_y), (dev_X, dev_y) = mnist
    return train_y


@pytest.fixture(scope="module")
def dev_X(mnist):
    (train_X, train_y), (dev_X, dev_y) = mnist
    return dev_X


@pytest.fixture(scope="module")
def dev_Y(mnist):
    (train_X, train_y), (dev_X, dev_y) = mnist
    return dev_y


def create_relu_softmax(depth, width):
    with Model.define_operators({"**": clone, ">>": chain}):
        model = ReLu(width) ** depth >> Softmax(10, width)
    return model


@pytest.fixture(params=[create_relu_softmax])
def create_model(request):
    return request.param


@pytest.mark.slow
@pytest.mark.parametrize(("depth", "width", "nb_epoch"), [(2, 32, 5)])
def test_small_end_to_end(
    depth, width, nb_epoch, create_model, train_X, train_Y, dev_X, dev_Y
):
    batch_size = 128
    model = create_model(depth, width).initialize(X=train_X[:10], Y=train_Y[:10])
    optimizer = Adam(0.001)
    losses = []
    scores = []
    for i in range(nb_epoch):
        for X, Y in model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True):
            Yh, backprop = model.begin_update(X)
            backprop(Yh - Y)
            model.finish_update(optimizer)
            losses.append(((Yh - Y) ** 2).sum())
        correct = 0
        total = 0
        for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
            Yh = model.predict(X)
            correct += (Yh.argmax(axis=0) == Y.argmax(axis=0)).sum()
            total += Yh.shape[0]
        score = correct / total
        scores.append(score)
    assert losses[-1] < losses[0]
    assert scores[-1] > scores[0]
