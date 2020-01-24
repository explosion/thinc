import pytest
from thinc.api import ReLu, Softmax, chain, clone, Adam
from thinc.api import PyTorchWrapper, TensorFlowWrapper
from thinc.util import has_torch, has_tensorflow
import ml_datasets


@pytest.fixture(scope="module")
def mnist(limit=5000):
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
    return (train_X[:limit], train_Y[:limit]), (dev_X[:limit], dev_Y[:limit])


def create_relu_softmax(depth, width, dropout, nI, nO):
    return chain(clone(ReLu(nO=width, dropout=dropout), depth), Softmax(10, width))


def create_wrapped_pytorch(depth, width, dropout, nI, nO):
    if not has_torch:
        pytest.skip(reason="Needs PyTorch")

    import torch
    import torch.nn
    import torch.nn.functional as F

    # TODO: rewrite to add depth
    class PyTorchModel(torch.nn.Module):
        def __init__(self, width, nO, nI, dropout):
            super(PyTorchModel, self).__init__()
            self.dropout1 = torch.nn.Dropout2d(dropout)
            self.dropout2 = torch.nn.Dropout2d(dropout)
            self.fc1 = torch.nn.Linear(nI, width)
            self.fc2 = torch.nn.Linear(width, nO)

        def forward(self, x):
            x = F.relu(x)
            x = self.dropout1(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    return PyTorchWrapper(PyTorchModel(width, nO, nI, dropout))


def create_wrapped_tensorflow(depth, width, dropout, nI, nO):
    if not has_tensorflow:
        pytest.skip(reason="Needs TensorFlow")

    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential

    tf_model = Sequential()
    for i in range(depth):
        tf_model.add(Dense(width, activation="relu", input_shape=(nI,)))
        tf_model.add(Dropout(dropout))
    tf_model.add(Dense(nO, activation="softmax"))
    return TensorFlowWrapper(tf_model)


@pytest.fixture(
    params=[create_relu_softmax, create_wrapped_pytorch, create_wrapped_tensorflow]
)
def create_model(request):
    return request.param


@pytest.mark.slow
@pytest.mark.parametrize(("depth", "width", "nb_epoch"), [(2, 32, 3)])
def test_small_end_to_end(depth, width, nb_epoch, create_model, mnist):
    batch_size = 128
    dropout = 0.2
    (train_X, train_Y), (dev_X, dev_Y) = mnist
    model = create_model(
        depth, width, dropout, nI=train_X.shape[1], nO=train_Y.shape[1]
    )
    model.initialize(X=train_X[:5], Y=train_Y[:5])
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
            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]
        score = correct / total
        scores.append(score)
    assert losses[-1] < losses[0]
    assert scores[-1] > scores[0]
    assert any([score > 0.2 for score in scores])
