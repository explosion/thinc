from __future__ import print_function
import plac
import pickle

from thinc.neural.vec2vec import ReLu, Softmax
from thinc.api import clone, chain
from thinc.neural.loss import categorical_crossentropy

from thinc.extra import datasets


def explicit_no_sugar_no_magic(depth, width, input_size, output_size,
        train_X, train_y, dev_X, dev_y):
    '''
    and train it without any shortcuts or helpers.

    This isn't what you'll do every day, but it shows the 'bones' of the library
    — how things are put together underneath.
    '''
    # In practice you'll use the Adam, SGD, etc classes. But, to be explicit...
    def optimizer(weights, gradient):
        weights += gradient * 0.001
        gradient.fill(0)
    # Repeat for N epochs
    for epoch in range(nb_epoch):
        # Shuffle indices, so we can make minibatches
        indices = list(range(len(train_X)))
        random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            # Get a minibatch of input (X),
            # and the matching labels (y).
            batch_indices = indices[i : i+batch_size]
            # Arrays decorated with shapes, to make it easier to follow.
            X__BO = train_X[batch_indices]
            y__B = train_y[batch_indices]
            # Start a backprop iteration:
            # Compute the forward pass, and get a callback to continue
            yh__BO, finish_update = model.begin_update(X__BO, dropout=0.2)
            # Use the scores to compute a scalar loss, and the gradient of loss.
            batch_loss, d_loss__BO = categorical_crossentropy(yh__BO, y__B)
            # Complete the backward pass.
            finish_update(d_loss__BO, optimizer)
            epoch_loss += batch_loss
        # At the end of the epoch, track the accuracy.
        acc = 0.
        dev_yh = model.predict(dev_X)
        for i in range(len(dev_y)):
            if dev_yh[i].argmax() == dev_y[i]:
                acc += 1
        print("Epoch", epoch, "loss", epoch_loss, "acc", acc)
        

def main(depth=2, width=512, nb_epoch=20):
    with Model.define_operators({'*': clone, '>>': chain}):
        model = ReLu(width) * depth >> Softmax()
    
    (train_X, train_Y), (dev_X, dev_Y), (test_X, test_Y) = datasets.mnist()

    with model.begin_training(train_X, train_Y) as trainer, optimizer:
        trainer.each_epoch(print_accuracy(dev_X, dev_y))
        for X, y in trainer.iterate(train_X, train_y):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            loss, d_loss = categorical_crossentropy(guess, yh)
            backprop(d_loss, optimizer(loss))
    with model.use_params(optimizer.averages):
        print('Avg dev.: %.3f' % model.evaluate(dev_X, dev_Y))
        print('Avg test.: %.3f' % model.evaluate(test_X, test_Y))
        with open('out.pickle', 'wb') as file_:
            pickle.dump(model, file_, -1)


##########################################################
# Example code, showing how to do things more explicitly.
##########################################################

def declare_sized_relu_mlp_explicit(depth, width, input_size, output_size):
    '''This function shows how to declare the model without helper functions.'''
    layers = [ReLu(width, input_size)]
    for _ in range(1, depth):
        layers.append(ReLu(width, width))
    layers.append(Softmax(output_size, width))
    model = FeedForward(layers)
    return model


def declare_sized_relu_mlp_with_helpers(depth, width, input_size, output_size):
    '''This function shows how to declare the model with the
    chain and clone helper functions, but with explicit
    input_size and output_size.'''
    layers = [ReLu(width, input_size)]
    layers.extend(clone(ReLu(width, width), depth))
    layers = [Softmax(output_size, width)]
    model = chain(*layers)
    return model


def declare_unsized_relu_mlp_with_chain_clone(depth, width):
    '''This function shows how to declare the model with the
    chain and clone helper functions. The input_size and output_size are left
    implicit. They'll be set once the data is available.'''
    model = chain(clone(ReLu(width), depth), Softmax())
    return model


def declare_unsized_relu_mlp_with_operators(depth, width):
    '''This function shows how to declare the model with the
    chain and clone helper functions temporarily bound to the '>>' and '**'
    operators.
    
    The input_size and output_size are left implicit. They'll be set once
    the data is available.'''
    with Model.define_operators({'>>': chain, '**': clone}):
        model = ReLu(width) ** depth >> Softmax()
    return model


def train_single_batch(model, optimizer, X__BO, y__BO, dropout=0.2):
    # Start a backprop iteration:
    # Compute the forward pass, and get a callback to continue
    yh__BO, finish_update = model.begin_update(X__BO, dropout)
    # Use the scores to compute a scalar loss, and the gradient of loss.
    batch_loss, d_loss__BO = categorical_crossentropy(yh__BO, y__B)
    # Complete the backward pass.
    finish_update(d_loss__BO, optimizer)
    return batch_loss


def evaluate_explicitly(model, dev_X, dev_y):
    # At the end of the epoch, track the accuracy.
    acc = 0.
    dev_yh = model.predict(dev_X)
    for i in range(len(dev_y)):
        if dev_yh[i].argmax() == dev_y[i]:
            acc += 1
    return acc
 

def minibatch(train_X, train_y, batch_size):
    # Shuffle indices, so we can make minibatches
    indices = list(range(len(train_X)))
    random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        # Get a minibatch of input (X),
        # and the matching labels (y).
        batch_indices = indices[i : i+batch_size]
        # Arrays decorated with shapes, to make it easier to follow.
        X__BO = train_X[batch_indices]
        y__B = train_y[batch_indices]
        yield X__BO, y__B
 

if __name__ == '__main__':
    plac.call(main)
