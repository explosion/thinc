import ml_datasets
import thinc
import tensorflow as tf

from thinc.layers.tensorflow_wrapper import TensorFlowWrapper
from tqdm import tqdm

try:
    import cupy as xp
    has_cupy = True
except ImportError:
    import numpy as xp
    has_cupy = False

# Tensorflow Hogs the entire GPU so using with cupy will cause OOM errors
# Thus make the GPU memory to grow in Tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def create_tf_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


def categorical_crossentropy(y_pred, y_true, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.

    Returns: scalar
    """
    y_pred = xp.clip(y_pred, epsilon, 1. - epsilon)  # for numerical stability
    ce = -xp.sum(y_true * xp.log(y_pred + 1e-9)) / len(y_pred)
    return ce


def main():
    batch_size = 128
    epochs = 10
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()

    if has_cupy:
        train_X = xp.asarray(train_X)
        train_Y = xp.asarray(train_Y)
        dev_X = xp.asarray(dev_X)
        dev_Y = xp.asarray(dev_Y)

    train_X /= 255
    dev_X /= 255
    print(train_X.shape[0], 'train samples')
    print(dev_X.shape[0], 'test samples')

    thinc_model = TensorFlowWrapper(create_tf_model())
    # this prints the tensorflow model
    print(thinc_model.shims[0])
    optimizer = thinc.optimizers.Adam(learn_rate=0.001)

    dev_predictions = thinc_model.predict(dev_X[:1000])
    dev_loss = categorical_crossentropy(y_pred=dev_predictions[:1000],
                                        y_true=dev_Y[:1000]
                                        )
    dev_accuracy = thinc.util.evaluate_model_on_arrays(thinc_model, dev_X[:1000],
                                                       dev_Y[:1000],
                                                       batch_size=batch_size)
    print("\nInitial results on a subset of dev set. loss: {}, accuracy: {}\n".format(dev_loss, dev_accuracy))
    # Notice that inital loss is around 2.302 which = ln(0.1).
    # This ensures that our implementation of loss function is correct

    # training Loop
    all_epoch_losses = []
    all_epoch_accuracy = []
    all_dev_loss = []
    all_dev_accuracy = []
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        train_generator = thinc.util.get_shuffled_batches(train_X, train_Y, batch_size=batch_size)
        epoch_loss = []
        epoch_accuracy = []
        for batch_x, batch_y in tqdm(train_generator, total=(len(train_X) // batch_size) + 1):
            predictions, backprop = thinc_model.begin_update(batch_x)
            loss = categorical_crossentropy(y_true=batch_y,
                                            y_pred=predictions
                                            )
            accuracy = thinc.util.evaluate_model_on_arrays(thinc_model, batch_x,
                                                           batch_y,
                                                           batch_size=batch_size)
            # reference https://deepnotes.io/softmax-crossentropy
            dloss_dpred = predictions - batch_y
            dX = backprop(dloss_dpred)
            thinc_model.finish_update(optimizer)

            epoch_loss.append(loss.mean())
            epoch_accuracy.append(accuracy.mean())

        mean_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        mean_epoch_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)

        all_epoch_losses.append(mean_epoch_loss)
        all_epoch_accuracy.append(mean_epoch_accuracy)

        # calculate performance on a subset of dev set
        dev_predictions = thinc_model.predict(dev_X[:1000])
        dev_loss = categorical_crossentropy(y_pred=dev_predictions,
                                            y_true=dev_Y[:1000]
                                            )
        dev_accuracy = thinc.util.evaluate_model_on_arrays(thinc_model, dev_X[:1000],
                                                           dev_Y[:1000],
                                                           batch_size=batch_size)
        all_dev_loss.append(dev_loss)
        all_dev_accuracy.append(accuracy)
        print("train_loss: {}, train_accuracy: {}, dev_loss: {}, dev_accuracy {}".format(mean_epoch_loss,
                                                                                         mean_epoch_accuracy,
                                                                                         dev_loss,
                                                                                         dev_accuracy))

    # Calculate final performance on a subset of dev. This will match with the last run
    dev_predictions = thinc_model.predict(dev_X[:1000])
    dev_loss = categorical_crossentropy(y_pred=dev_predictions,
                                        y_true=dev_Y[:1000]
                                        )
    dev_accuracy = thinc.util.evaluate_model_on_arrays(thinc_model, dev_X[:1000],
                                                       dev_Y[:1000],
                                                       batch_size=batch_size)
    print("\nfinal results on a subset of dev set. loss: {}, accuracy: {}".format(dev_loss, dev_accuracy))


if __name__ == "__main__":
    main()
