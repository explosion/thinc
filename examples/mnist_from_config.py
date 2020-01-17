# pip install thinc ml_datasets tqdm typer
from typing import Optional
from pathlib import Path
from thinc.api import minibatch, evaluate_model_on_arrays, Config
import thinc
import ml_datasets
from wasabi import msg
import tqdm
import typer


CONFIG = """
[hyper_params]
n_hidden = 512
dropout = 0.2
learn_rate = 0.001

[model]
@layers = "chain.v0"

[model.*.relu1]
@layers = "ReLu.v0"
nO = ${hyper_params:n_hidden}
dropout = ${hyper_params:dropout}

[model.*.relu2]
@layers = "ReLu.v0"
nO = ${hyper_params:n_hidden}
dropout = ${hyper_params:dropout}

[model.*.softmax]
@layers = "Softmax.v0"

[optimizer]
@optimizers = "Adam.v1"
learn_rate = ${hyper_params:learn_rate}
"""


def main(config_path: Optional[Path] = None, n_iter: int = 10, batch_size: int = 128):
    # You can edit the CONFIG string within the file, or copy it out to
    # a separate file and pass in the path.
    if config_path is None:
        config = Config().from_str(CONFIG)
    else:
        config = Config().from_disk(config_path)
    print(config)
    loaded_config = thinc.registry.make_from_config(config)
    # Here we have the model and optimizer, built for us by the registry.
    model = loaded_config["model"]
    optimizer = loaded_config["optimizer"]

    # Load the data
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
    # Set any missing shapes for the model.
    model.initialize(X=train_X[:5], Y=train_Y[:5])
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
