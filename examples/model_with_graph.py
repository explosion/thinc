from thinc.layers import chain, ReLu, Softmax, Affine, ExtractWindow, Maxout
import ml_datasets
import typer


CONFIG = """
[hyper_params]
n_hidden = 512
dropout = 0.2

[model]
@layers = "chain.v1"

[model.layers.relu1]
@layers = "ReLu.v1"
nO = ${hyper_params:n_hidden}
dropout = ${hyper_params:dropout}

[model.layers.relu2]
@layers = "ReLu.v1"
nO = ${hyper_params:n_hidden}
dropout = ${hyper_params:dropout}

[model.layers.softmax]
@layers = "Softmax.v1"

[optimizer]
@optimizers = "Adam.v1"
learn_rate = ${hyper_params:learn_rate}
"""


def load_mnist():
    from thinc.backends import NumpyOps
    from thinc.util import to_categorical

    ops = NumpyOps()
    mnist_train, mnist_dev, _ = ml_datasets.mnist()
    train_X, train_Y = ops.unzip(mnist_train)
    train_Y = to_categorical(train_Y, nb_classes=10)
    dev_X, dev_Y = ops.unzip(mnist_dev)
    dev_Y = to_categorical(dev_Y, nb_classes=10)
    return (train_X, train_Y), (dev_X, dev_Y)


def main(
    n_hidden: int = 32, dropout: float = 0.2, n_iter: int = 10, batch_size: int = 128
):
    # Define the model
    model = chain(
        ExtractWindow(3),
        ReLu(n_hidden, dropout=dropout, normalize=True),
        Maxout(n_hidden * 4),
        Affine(n_hidden * 2),
        ReLu(n_hidden, dropout=dropout, normalize=True),
        Affine(n_hidden),
        ReLu(n_hidden, dropout=dropout),
        Softmax(),
    )

    # Load the data
    (train_X, train_Y), (dev_X, dev_Y) = load_mnist()
    # Set any missing shapes for the model.
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    model.visualize(output="tmp/model.svg")


if __name__ == "__main__":
    typer.run(main)
