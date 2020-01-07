import ml_datasets
import thinc
import tensorflow as tf
import numpy as np

from thinc.layers.tensorflow_wrapper import TensorFlowWrapper
from tqdm import tqdm

batch_size = 128
epochs = 10


def main():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    print(model.summary())

    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()

    train_X /= 255
    dev_X /= 255
    print(train_X.shape[0], 'train samples')
    print(dev_X.shape[0], 'test samples')

    thinc_model = TensorFlowWrapper(model)
    optimizer = thinc.optimizers.Adam(learn_rate=0.001)

    dev_predictions = thinc_model.predict(dev_X[:1000])
    dev_loss = tf.keras.losses.categorical_crossentropy(y_true=thinc.util.xp2tensorflow(dev_Y[:1000]),
                                                        y_pred=thinc.util.xp2tensorflow(dev_predictions)
                                                        ).numpy().mean()
    dev_accuracy = tf.keras.metrics.categorical_accuracy(y_true=thinc.util.xp2tensorflow(dev_Y[:1000]),
                                                         y_pred=thinc.util.xp2tensorflow(dev_predictions)
                                                         ).numpy().mean()
    print("\nInitial results on a subset of dev set. loss: {}, accuracy: {}\n".format(dev_loss, dev_accuracy))

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
        for batch_x, batch_y in tqdm(train_generator, total=len(train_X) // batch_size):
            predictions, backprop = thinc_model.begin_update(batch_x)
            # convert predictions to Tensorflow tensors
            predictions = thinc.util.xp2tensorflow(predictions)
            # calculate d_loss/d_predictions
            with tf.GradientTape() as tape:
                tape.watch(predictions)
                loss = tf.keras.losses.categorical_crossentropy(y_true=thinc.util.xp2tensorflow(batch_y),
                                                                y_pred=predictions
                                                                )
            accuracy = tf.keras.metrics.categorical_accuracy(y_true=thinc.util.xp2tensorflow(batch_y),
                                                             y_pred=predictions
                                                             )
            dloss_dpred = tape.gradient(loss, predictions)
            dX = backprop(dloss_dpred)
            thinc_model.finish_update(optimizer)

            epoch_loss.append(loss.numpy().mean())
            epoch_accuracy.append(accuracy.numpy().mean())

        mean_epoch_loss = np.mean(epoch_loss)
        mean_epoch_accuracy = np.mean(epoch_accuracy)

        all_epoch_losses.append(mean_epoch_loss)
        all_epoch_accuracy.append(mean_epoch_accuracy)

        # calculate performance on a subset of dev set
        dev_predictions = thinc_model.predict(dev_X[:1000])
        dev_loss = tf.keras.losses.categorical_crossentropy(y_true=thinc.util.xp2tensorflow(dev_Y[:1000]),
                                                            y_pred=thinc.util.xp2tensorflow(dev_predictions)
                                                            ).numpy().mean()
        dev_accuracy = tf.keras.metrics.categorical_accuracy(y_true=thinc.util.xp2tensorflow(dev_Y[:1000]),
                                                             y_pred=thinc.util.xp2tensorflow(dev_predictions)
                                                             ).numpy().mean()
        all_dev_loss.append(dev_loss)
        all_dev_accuracy.append(accuracy)
        print("train_loss: {}, train_accuracy: {}, dev_loss: {}, dev_accuracy {}".format(mean_epoch_loss,
                                                                                         mean_epoch_accuracy,
                                                                                         dev_loss,
                                                                                         dev_accuracy))

    dev_predictions = thinc_model.predict(dev_X[:1000])
    dev_loss = tf.keras.losses.categorical_crossentropy(y_true=thinc.util.xp2tensorflow(dev_Y[:1000]),
                                                        y_pred=thinc.util.xp2tensorflow(dev_predictions)
                                                        ).numpy().mean()
    dev_accuracy = tf.keras.metrics.categorical_accuracy(y_true=thinc.util.xp2tensorflow(dev_Y[:1000]),
                                                         y_pred=thinc.util.xp2tensorflow(dev_predictions)
                                                         ).numpy().mean()
    print("\nfinal results on a subset of dev set. loss: {}, accuracy: {}".format(dev_loss, dev_accuracy))


if __name__ == "__main__":
    main()
