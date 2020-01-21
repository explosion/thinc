"""
PyTorch version: https://github.com/pytorch/examples/blob/master/mnist/main.py
TensorFlow version: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
"""
# pip install thinc ml_datasets typer
from thinc.api import Model, chain, ReLu, Softmax, Adam, minibatch, Pairs
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
        ReLu(nO=n_hidden, dropout=dropout),
        ReLu(nO=n_hidden, dropout=dropout),
        Softmax(),
    )
    # Load the data
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
    # Set any missing shapes for the model.
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    # Create the optimizer.
    optimizer = Adam(0.001)
    for i in range(n_iter):
        for X, Y in model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True):
            Yh, backprop = model.begin_update(X)
            backprop(Yh - Y)
            model.finish_update(optimizer)
        # Evaluate and print progress
        correct = 0
        total = 0
        for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
            Yh = model.predict(X)
            corrct += (Yh.argmax(axis=0) == Y.argmax(axis=0)).sum()
            total += Yh.shape[0]
        score = correct / total
        msg.row((i, f"{score:.3f}"), widths=(3, 5))


if __name__ == "__main__":
    typer.run(main)
