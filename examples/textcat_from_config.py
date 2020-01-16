import random
from typing import Optional, List
from pathlib import Path
import thinc
from thinc.api import fix_random_seed, Adam, Model, Config
from thinc.layers import chain, list2ragged, with_array, MeanPool, Softmax
from thinc.layers import Embed
from thinc.types import Array2d
from wasabi import msg
import typer
import numpy as np
import csv
from ml_datasets import loaders
from syntok.tokenizer import Tokenizer


CONFIG = """
[hyper_params]
width = 64

[model]
@layers = "EmbedPoolTextcat.v0"

[model.embed]
@layers = "Embed.v0"
nO = ${hyper_params:width}

[optimizer]
@optimizers = "Adam.v1"
learn_rate = 0.001

[training]
batch_size = 8
n_iter = 10
"""


def main(
    config_path: Optional[Path] = None,
    n_examples: Optional[int] = 2000,
    dataset: Optional[str] = "dbpedia",
):
    fix_random_seed(0)

    # Load data
    supported_datasets = ["dbpedia", "imdb"]
    if dataset not in supported_datasets:
        msg.fail("Supported datasets:" + ", ".join(supported_datasets), exits=1)
    msg.text(f"Loading dataset '{dataset}'...")
    dataset_loader = loaders.get(dataset)
    train_data, dev_data = dataset_loader(limit=n_examples)
    train_texts, train_cats = zip(*train_data)
    dev_texts, dev_cats = zip(*dev_data)
    unique_cats = list(np.unique(np.concatenate((train_cats, dev_cats))))
    nr_class = len(unique_cats)
    msg.text(f"  {len(train_data)} training instances")
    msg.text(f"  {len(dev_data)} dev instances")
    msg.text(f"  {nr_class} classes")

    train_y = np.zeros((len(train_cats), nr_class), dtype="f")
    for i, cat in enumerate(train_cats):
        train_y[i][unique_cats.index(cat)] = 1
    dev_y = np.zeros((len(dev_cats), nr_class), dtype="f")
    for i, cat in enumerate(dev_cats):
        dev_y[i][unique_cats.index(cat)] = 1

    # Tokenize texts
    train_tokenized = tokenize_texts(train_texts)
    dev_tokenized = tokenize_texts(dev_texts)

    # Generate simple vocab mapping, <unk> is 0
    vocab = {}
    count_id = 1
    for text in train_tokenized:
        for token in text:
            if token not in vocab:
                vocab[token] = count_id
                count_id += 1

    # Map texts using vocab
    train_X = []
    for text in train_tokenized:
        train_X.append(np.array([vocab.get(t, 0) for t in text]))
    dev_X = []
    for text in dev_tokenized:
        dev_X.append(np.array([vocab.get(t, 0) for t in text]))

    # You can edit the CONFIG string within the file, or copy it out to
    # a separate file and pass in the path.
    if config_path is None:
        config = Config().from_str(CONFIG)
    else:
        config = Config().from_disk(config_path)

    # Load the config
    loaded_config = thinc.registry.make_from_config(config)

    # Here we have the model and optimizer, built for us by the registry.
    model = loaded_config["model"]
    model.get_ref("embed").set_dim("nV", len(vocab))
    model.initialize(X=train_X, Y=train_y)

    optimizer = loaded_config["optimizer"]

    # Get training parameters from config
    batch_size = config["training"]["batch_size"]
    n_iter = config["training"]["n_iter"]

    # Train
    msg.text("Training...")
    row_widths = (4, 8, 8)
    msg.row(("Iter", "Loss", f"Accuracy"), widths=row_widths)
    msg.row("-"*width for width in row_widths)
    zipped = list(zip(train_X, train_y))
    for n in range(n_iter):
        loss = 0.0
        random.shuffle(zipped)
        for i in range(0, len(zipped), batch_size):
            X, Y = zip(*zipped[i : i + batch_size])
            Yh, backprop = model.begin_update(X)
            d_loss = []
            for i in range(len(Yh)):
                d_loss.append(Yh[i] - Y[i])
                loss += ((Yh[i] - Y[i]) ** 2).sum()
            backprop(np.array(d_loss))
            model.finish_update(optimizer)
        score = evaluate_textcat(model, dev_X, dev_y, batch_size)
        msg.row((n, f"{loss:.2f}", f"{score:.3f}"), widths=row_widths)


@thinc.registry.layers("EmbedPoolTextcat.v0")
def EmbedPoolTextcat(embed: Model[Array2d, Array2d]) -> Model[List[Array2d], Array2d]:
    model = chain(
        list2ragged(),
        with_array(embed),
        MeanPool(),
        Softmax()
    )
    model.set_ref("embed", embed)
    return model


def evaluate_textcat(model, dev_X, dev_Y, batch_size):
    correct = 0.0
    total = 0.0
    for i in range(0, len(dev_X), batch_size):
        Yh = model.predict(dev_X[i : i + batch_size])
        Y = dev_Y[i : i + batch_size]
        for j in range(len(Yh)):
            correct += Yh[j].argmax(axis=0) == Y[j].argmax(axis=0)
        total += len(Y)
    return correct / total


def tokenize_texts(texts):
    tok = Tokenizer()
    return [[token.value for token in tok.tokenize(text)] for text in texts]


if __name__ == "__main__":
    typer.run(main)
