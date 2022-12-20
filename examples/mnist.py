"""
PyTorch version: https://github.com/pytorch/examples/blob/master/mnist/main.py
TensorFlow version: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
"""
# pip install thinc ml_datasets typer
from thinc.api import Model, chain, Relu, Softmax, Adam
from thinc.api import CategoricalCrossentropy
import ml_datasets
from wasabi import msg
from tqdm import tqdm
import typer


def main(
    n_hidden: int = 256, dropout: float = 0.2, n_iter: int = 10, batch_size: int = 128
):
    # Define the model
    model: Model = chain(
        Relu(nO=n_hidden, dropout=dropout),
        Relu(nO=n_hidden, dropout=dropout),
        Softmax(),
    )
    # Load the data
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
    loss_func = CategoricalCrossentropy()
    # Set any missing shapes for the model.
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    train_data = model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True)
    dev_data = model.ops.multibatch(batch_size, dev_X, dev_Y)
    # Create the optimizer.
    optimizer = Adam(0.001)
    for i in range(n_iter):
        for X, Y in tqdm(train_data, leave=False):
            Yh, backprop = model.begin_update(X)
            grad, loss = loss_func(Yh, Y)
            backprop(grad)
            model.finish_update(optimizer)
        # Evaluate and print progress
        correct = 0
        total = 0
        for X, Y in dev_data:
            Yh = model.predict(X)
            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]
        score = correct / total
        msg.row((i, f"{score:.3f}"), widths=(3, 5))


if __name__ == "__main__":
    typer.run(main)
