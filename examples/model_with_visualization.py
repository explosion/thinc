from thinc.api import chain, ReLu, Softmax, Linear, ExtractWindow, Maxout, Model
from thinc.extra.visualizers import pydot_visualizer
import ml_datasets
import typer


def main(
    n_hidden: int = 32,
    dropout: float = 0.2,
    n_iter: int = 10,
    batch_size: int = 128,
    output: str = "model.svg",
    file_format: str = "svg",
):
    # Define the model
    model: Model = chain(
        ExtractWindow(3),
        ReLu(n_hidden, dropout=dropout, normalize=True),
        Maxout(n_hidden * 4),
        Linear(n_hidden * 2),
        ReLu(n_hidden, dropout=dropout, normalize=True),
        Linear(n_hidden),
        ReLu(n_hidden, dropout=dropout),
        Softmax(),
    )
    # Load the data
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
    # Set any missing shapes for the model.
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    pydot_visualizer(model, output=output, file_format=file_format)


if __name__ == "__main__":
    typer.run(main)
