# TODO: This example currently doesn't work due to how pydantic validates
# generics in fields (layers expecting varialbe arguments annotated as
# Model[InT, OutT]). The example also needs to be rewritten to define the model
# in a custom registered function instead of only in the config. This is
# closer to the recommended best practice.
from typing import Optional
from pathlib import Path
import random
import thinc
from thinc.api import Config, fix_random_seed
from wasabi import msg
import ml_datasets
import typer


CONFIG = """
[hyper_params]
depth = 2
width = 32
vector_width = 16
learn_rate = 0.001

[model]
@layers = "chain.v0"

[model.*.strings2arrays]
@layers = "strings2arrays.v0"

[model.*.with_array]
@layers = "with_array.v0"

[model.*.with_array.layer]
@layers = "chain.v0"

[model.*.with_array.layer.*.hashembed]
@layers = "HashEmbed.v0"
nO = ${hyper_params:width}
nV = ${hyper_params:vector_width}

[model.*.with_array.layer.*.extractwindow]
@layers = "ExtractWindow.v0"
window_size = 1

[model.*.with_array.layer.*.relu1]
@layers = "ReLu.v0"
nO = ${hyper_params:width}
nI = 48

[model.*.with_array.layer.*.relu2]
@layers = "ReLu.v0"
nO = ${hyper_params:width}
nI = ${hyper_params:vector_width}

[model.*.with_array.layer.*.softmax]
@layers = "Softmax.v0"
nO = 17
nI = ${hyper_params:vector_width}

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

    fix_random_seed(0)
    (train_X, train_y), (dev_X, dev_y) = ml_datasets.ud_ancora_pos_tags()

    for n in range(n_iter):
        loss = 0.0
        zipped = list(zip(train_X, train_y))
        random.shuffle(zipped)
        for i in range(0, len(zipped), batch_size):
            X, Y = zip(*zipped[i : i + batch_size])
            Yh, backprop = model.begin_update(X)
            d_loss = []
            for i in range(len(Yh)):
                d_loss.append(Yh[i] - Y[i])
                loss += ((Yh[i] - Y[i]) ** 2).sum()
            backprop(d_loss)
            model.finish_update(optimizer)
        score = evaluate_tagger(model, dev_X, dev_y, batch_size)
        msg.row((n, f"{loss:.2f}", f"{score:.3f}"), widths=(3, 8, 5))


def evaluate_tagger(model, dev_X, dev_Y, batch_size):
    correct = 0.0
    total = 0.0
    for i in range(0, len(dev_X), batch_size):
        Yh = model.predict(dev_X[i : i + batch_size])
        Y = dev_Y[i : i + batch_size]
        for j in range(len(Yh)):
            correct += (Yh[j].argmax(axis=1) == Y[j].argmax(axis=1)).sum()
            total += Yh[j].shape[0]
    return correct / total


if __name__ == "__main__":
    typer.run(main)
