"""
PyTorch version: https://github.com/pytorch/examples/blob/master/mnist/main.py
TensorFlow version: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
"""
# pip install thinc ml_datasets typer
from thinc.api import Model, chain, ReLu, Softmax, Adam, minibatch
from thinc.api import evaluate_model_on_arrays
import ml_datasets
from wasabi import msg
import tqdm
import typer


def main(
    n_hidden: int = 32, dropout: float = 0.2, n_iter: int = 10, batch_size: int = 128
):
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
    # Train the model
    indices = model.ops.xp.arange(train_X.shape[0], dtype="i")
    for i in range(n_iter):
        model.ops.xp.random.shuffle(indices)
        for idx_batch in minibatch(tqdm.tqdm(indices, leave=False)):
            Yh, backprop = model.begin_update(train_X[idx_batch])
            backprop(Yh - train_Y[idx_batch])
            model.finish_update(optimizer)
        # Evaluate and print progress
        score = evaluate_model_on_arrays(model, dev_X, dev_Y, batch_size=batch_size)
        msg.row((i, f"{score:.3f}"), widths=(3, 5))


if __name__ == "__main__":
    typer.run(main)
