import tensorflow as tf
import ml_datasets
import tqdm

from thinc.api import prefer_gpu  # noqa: F401
from thinc.api import Adam, TensorflowWrapper
from thinc.api import get_shuffled_batches, evaluate_model_on_arrays, get_array_module
from thinc.api import get_current_ops


def create_tf_model(width, depth, *, nO=10, nI=784):
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential()
    for _ in range(depth):
        model.add(Dense(width, activation="relu", input_shape=(nI,)))
        model.add(Dropout(0.2))
    model.add(Dense(nO, activation="softmax"))
    return model


def main(
    width: int = 512,
    depth: int = 2,
    dropout: float = 0.2,
    batch_size: int = 128,
    epochs: int = 10,
):
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()

    print(train_X.shape[0], "train samples", dev_X.shape[0], "test samples")

    model = TensorflowWrapper(create_tf_model(width, depth))
    optimizer = Adam(learn_rate=0.001)

    n_batches = (len(train_X) // batch_size) + 1
    for epoch in range(epochs):
        train_generator = get_shuffled_batches(train_X, train_Y, batch_size=batch_size)
        for inputs, truths in tqdm.tqdm(train_generator, total=n_batches, leave=False):
            guesses, backprop = model.begin_update(inputs)
            d_guesses = guesses - truths
            d_inputs = backprop(d_guesses)
            model.finish_update(optimizer)

        accuracy = evaluate_model_on_arrays(model, dev_X, dev_Y, batch_size=batch_size)
        print(epoch, accuracy)


if __name__ == "__main__":
    import typer

    typer.run(main)
