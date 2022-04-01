"""
PyTorch version: https://github.com/pytorch/examples/blob/master/mnist/main.py
TensorFlow version: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
"""
# pip install thinc ml_datasets typer
from thinc.api import Model, chain, Relu, Adam, Softmax
import ml_datasets
from wasabi import msg
from tqdm import tqdm
import typer
from thinc.optimizers import Lookahead, SWA
from thinc.schedules import cyclic_triangular


def main(
    n_hidden: int = 256, dropout: float = 0.2, n_iter: int = 10, batch_size: int = 128
):
    # Define the model
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist('fashion')
    model: Model = chain(
        Relu(nO=n_hidden, dropout=dropout),
        Relu(nO=n_hidden, dropout=dropout),
        Softmax(),
    )
    # Set any missing shapes for the model.
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    train_data = model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True)
    dev_data = model.ops.multibatch(batch_size, dev_X, dev_Y)
    # Create the optimizer.
    inner_optimizer = Adam(0.001)
    optimizer = Lookahead(inner_optimizer, freq=5, pullback=0.5)
    for i in range(n_iter):
        for X, Y in tqdm(train_data, leave=False):
            Yh, backprop = model.begin_update(X)
            backprop(Yh - Y)
            model.finish_update(optimizer)
        correct = 0
        total = 0
        # Turn off Lookahead and start SWA
        if i == 5:
            period = 20
            cyclic_schedule = cyclic_triangular(
                min_lr=0.0001, max_lr=0.002, period=period
            )
            optimizer = SWA(
                optimizer.optimizer, lr=cyclic_schedule, freq=period
            )
            optimizer.start_swa()
        # Use averages for evaluation
        with model.use_params(optimizer.averages):
            for X, Y in dev_data:
                Yh = model.predict(X)
                correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
                total += Yh.shape[0]
                score = correct / total
            msg.row((i, f"{score:.3f}"), widths=(3, 5))


if __name__ == "__main__":
    typer.run(main)
