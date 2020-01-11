import random
from thinc.api import Model, ReLu, Softmax, HashEmbed, ExtractWindow, chain
from thinc.api import with_array, strings2arrays, Adam, fix_random_seed
from wasabi import msg
import ml_datasets
import typer


def main(
    batch_size: int = 8,
    depth: int = 2,
    width: int = 32,
    vector_width: int = 16,
    n_iter: int = 5,
):
    fix_random_seed(0)
    (train_X, train_y), (dev_X, dev_y) = ml_datasets.ud_ancora_pos_tags()

    with Model.define_operators({">>": chain}):
        model = strings2arrays() >> with_array(
            HashEmbed(width, vector_width)
            >> ExtractWindow(window_size=1)
            >> ReLu(width, width * 3)
            >> ReLu(width, width)
            >> Softmax(17, width)
        )

    optimizer = Adam(0.001)
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
