import pytest
from thinc.model import Model
from thinc.layers.relu import ReLu
from thinc.layers.softmax import Softmax
from thinc.layers.embed import Embed
from thinc.layers.extractwindow import ExtractWindow
from thinc.layers.chain import chain
from thinc.layers.with_flatten import with_flatten
from thinc.loss import categorical_crossentropy
import ml_datasets


@pytest.fixture(scope="module")
def ancora():
    train_data, check_data, nr_class = ml_datasets.ancora_pos_tags()
    train_X, train_y = zip(*train_data)
    dev_X, dev_y = zip(*check_data)
    return (train_X[:100], train_y[:100]), (dev_X, dev_y)


@pytest.fixture(scope="module")
def train_X(ancora):
    (train_X, train_y), (dev_X, dev_y) = ancora
    return train_X


@pytest.fixture(scope="module")
def train_y(ancora):
    (train_X, train_y), (dev_X, dev_y) = ancora
    return train_y


@pytest.fixture(scope="module")
def dev_X(ancora):
    (train_X, train_y), (dev_X, dev_y) = ancora
    return dev_X


@pytest.fixture(scope="module")
def dev_y(ancora):
    (train_X, train_y), (dev_X, dev_y) = ancora
    return dev_y


def create_embed_relu_relu_softmax(depth, width, vector_length):
    with Model.define_operators({">>": chain}):
        model = with_flatten(
            Embed(width, vector_length)
            >> ExtractWindow(nW=1)
            >> ReLu(width)
            >> ReLu(width)
            >> Softmax(20)
        )
    return model


@pytest.fixture(params=[create_embed_relu_relu_softmax])
def create_model(request):
    return request.param


@pytest.mark.xfail
@pytest.mark.parametrize(
    ("depth", "width", "vector_width", "nb_epoch"), [(2, 32, 16, 5)]
)
def test_small_end_to_end(
    depth, width, vector_width, nb_epoch, create_model, train_X, train_y, dev_X, dev_y
):
    model = create_model(depth, width, vector_width)
    assert isinstance(model, Model)
    losses = []
    with model.begin_training(train_X, train_y) as (trainer, optimizer):
        trainer.nb_epoch = 10
        trainer.batch_size = 8
        for X, y in trainer.iterate(train_X, train_y):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            d_loss = []
            loss = []
            for i in range(len(yh)):
                dl, l = categorical_crossentropy(yh[i], y[i])  # noqa: E741
                d_loss.append(dl)
                loss.append(l)
            backprop(d_loss, optimizer)
            losses.append(sum(loss))
    assert losses[-1] < losses[0]
