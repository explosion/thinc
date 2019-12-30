from thinc.layers import chain, ReLu, Softmax
from thinc.optimizers import Adam
from thinc.util import minibatch, to_categorical
import ml_datasets
import tqdm


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


def main(n_hidden=32, dropout=0.2, n_iter=10):
    # Define the model
    model = chain(
        ReLu(n_hidden, dropout=dropout),
        ReLu(n_hidden, dropout=dropout),
        Softmax()
    )

    # Load the data
    mnist_train, mnist_dev, _ = ml_datasets.mnist()
    train_X, train_Y = model.ops.unzip(mnist_train)
    train_Y = to_categorical(train_Y, nb_classes=10)
    dev_X, dev_Y = model.ops.unzip(mnist_dev)
    dev_Y = to_categorical(dev_Y, nb_classes=10)

    # Set any missing shapes for the model.
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    # Create the optimizer.
    optimizer = Adam(0.001)
    
    # Train
    indices = model.ops.xp.arange(train_X.shape[0], dtype="i")
    for i in range(n_iter):
        model.ops.xp.random.shuffle(indices)
        for idx_batch in minibatch(tqdm.tqdm(indices, leave=False)):
            Yh, backprop = model.begin_update(train_X[idx_batch])
            backprop(Yh - train_Y[idx_batch])
            model.finish_update(optimizer)
        # Print progress


if __name__ == "__main__":
    import plac
    plac.call(main)
