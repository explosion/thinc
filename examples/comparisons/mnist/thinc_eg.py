from thinc.layers import chain, ReLu, Softmax
from thinc.optimizers import Adam
from thinc.util import get_shuffled_batches
import ml_datasets
import tqdm
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
        ReLu(n_hidden, dropout=dropout), ReLu(n_hidden, dropout=dropout), Softmax()
    )
    # Load the data
    (train_X, train_Y), (dev_X, dev_Y) = load_mnist()
    # Set any missing shapes for the model.
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    # Create the optimizer.
    optimizer = Adam(0.001, L2=0.0, use_averages=False, max_grad_norm=0.0)
    # Train
    for i in range(n_iter):
        train_batches = list(get_shuffled_batches(train_X, train_Y, batch_size))
        for images, truths in tqdm.tqdm(train_batches):

            guesses, backprop = model.begin_update(images)
            d_guesses = (guesses - truths) / guesses.shape[0]
            backprop(d_guesses)
            model.finish_update(optimizer)


if __name__ == "__main__":
    typer.run(main)
