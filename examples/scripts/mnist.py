from thinc.api import Model, chain, ReLu, Softmax, Adam, minibatch
from thinc.api import set_current_ops, JaxOps
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


def main(n_hidden: int = 32, dropout: float = 0.2, n_iter: int = 10):
    set_current_ops(JaxOps())
    # Define the model
    model: Model = chain(
        ReLu(n_hidden, dropout=dropout), ReLu(n_hidden, dropout=dropout), Softmax()
    )

    # Load the data
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
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
    typer.run(main)
