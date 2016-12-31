Thinc: Practical Machine Learning for NLP in Python
***************************************************

**Thinc** is the machine learning library powering `spaCy <https://spacy.io>`_. 
It features a battle-tested linear model designed for large sparse learning 
problems, and a flexible neural network model under development for
`spaCy v2.0 <https://github.com/explosion/spaCy/projects/3>`_.

Thinc is a practical toolkit for implementing models that follow the  
`"Embed, encode, attend, predict" <https://explosion.ai/blog/deep-learning-formula-nlp>`_ 
architecture. It's designed to be easy to install, efficient for CPU usage and
optimised for NLP and deep learning with text – in particular, hierarchically 
structured input and variable-length sequences.

.. image:: https://travis-ci.org/explosion/thinc.svg?branch=master
    :target: https://travis-ci.org/explosion/thinc
    :alt: Build Status

.. image:: https://img.shields.io/github/release/explosion/thinc.svg
    :target: https://github.com/explosion/thinc/releases   
    :alt: Current Release Version

.. image:: https://img.shields.io/pypi/v/thinc.svg   
    :target: https://pypi.python.org/pypi/thinc
    :alt: pypi Version

Quickstart
==========

If you have `Fabric <http://www.fabfile.org>`_ installed, you can use the shortcut:

.. code:: bash

   git clone https://github.com/explosion/thinc
   cd thinc
   fab clean env make test

You can then run the examples as follows:

.. code:: bash

   fab eg.mnist
   fab eg.basic_tagger

Otherwise, you can build and test explicitly with:

.. code:: bash

   git clone https://github.com/explosion/thinc
   cd thinc
   
   virtualenv .env
   source .env/bin/activate
   
   pip install -r requirements.txt
   python setup.py build_ext --inplace
   py.test thinc/

And then run the examples as follows:

.. code:: bash

   python examples/mnist.py
   python examples/basic_tagger.py

Design
======

Thinc is implemented in pure Python at the moment, using `Chainer <http://chainer.org/>`_'s cupy for GPU and numpy for CPU computations. Thinc doesn't use autodifferentiation. Instead, we just use callbacks.

Let's say you have a batch of data, of shape ``(B, I)``. You want to use this to update a model. To do that, you need to compute the model's output for that input, and also the gradient with respect to that output. Like so:

.. code:: python

    x__BO, finish_update = model.begin_update(x__BI)
    dx__BO = compute_gradient(dx__BO, y__B)
    dx__BI = finish_update(dx__BO)

To backprop through multiple layers, we simply accumulate the callbacks:

.. code:: python

    class Chain(list):
        def predict(self, X):
            for layer in self:
                X = layer(X)
            return X

        def begin_update(self, X, dropout=0.0):
            callbacks = []
            for layer in self.layers:
                X, callback = layer.begin_update(X, dropout=dropout)
            callbacks.append(callback)

            def finish_update(gradient, optimizer):
                for backprop in reversed(callbacks):
                    gradient = backprop(gradient, optimizer)
                return gradient
            return X, finish_update

The differentiation rules are pretty easy to work with, so long as every layer is a good citizen.

Adding layers
-------------

To add layers, you usually implement a subclass of ``base.Model`` or ``base.Network``. Use ``Network`` for layers which don't own weights data directly, but instead, chain together a sequence of models.

.. code:: python

    class ReLuMLP(Network):
        Hidden = ReLu
        Output = Softmax
        width = 128
        depth = 3

        def setup(self, nr_out, nr_in, **kwargs):
            for i in range(self.depth):
                self.layers.append(self.Hidden(nr_out=self.width, nr_in=nr_in,
                    name='hidden-%d' % i))
                nr_in = self.width
            self.layers.append(self.Output(nr_out=nr_out, nr_in=nr_in))
            self.set_weights(initialize=True)
            self.set_gradient()



When you implement a layer, there are two simple rules to follow to make sure it's well-behaved:

1. **Don't add side-effects to** ``begin_update``. Aside from the obvious concurrency problems, it's not nice to make the API silently produce incorrect results if the user calls the functions out of order.


2. **Keep the interfaces to** ``begin_update`` **and** ``finish_update`` **uniform**. We want to write generic functions to sum, concatenate, average, etc different layers. If your layer has a special interface, those generic functions won't work.
