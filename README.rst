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

🔮 **Version 6.3 out now!** `Read the release notes here. <https://github.com/explosion/thinc/releases/>`_

.. image:: https://travis-ci.org/explosion/thinc.svg?branch=master
    :target: https://travis-ci.org/explosion/thinc
    :alt: Build Status

.. image:: https://img.shields.io/coveralls/explosion/thinc.svg
    :target: https://coveralls.io/github/explosion/thinc
    :alt: Test Coverage

.. image:: https://img.shields.io/github/release/explosion/thinc.svg
    :target: https://github.com/explosion/thinc/releases   
    :alt: Current Release Version

.. image:: https://img.shields.io/pypi/v/thinc.svg   
    :target: https://pypi.python.org/pypi/thinc
    :alt: pypi Version
   
.. image:: https://img.shields.io/badge/gitter-join%20chat%20%E2%86%92-7676d1.svg
    :target: https://gitter.im/explosion/thinc
    :alt: Thinc on Gitter

.. image:: https://img.shields.io/twitter/follow/explosion_ai.svg?style=social&label=Follow
    :target: https://twitter.com/explosion_ai
    :alt: Follow us on Twitter

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
   fab eg.cnn_tagger

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
   python examples/cnn_tagger.py


Usage
=====

The Neural Network API is still subject to change, even within minor versions.
You can get a feel for the current API by checking out the examples. Here are
a few quick highlights.

1. Shape inference
------------------

Models can be created with some dimensions unspecified. Missing dimensions are
inferred when pre-trained weights are loaded or when training begins. This
eliminates a common source of programmer error:

.. code:: python

    # Invalid network — shape mismatch
    model = FeedForward(ReLu(512, 748), ReLu(512, 784), Softmax(10))
    
    # Leave the dimensions unspecified, and you can't be wrong.
    model = FeedForward(ReLu(512), ReLu(512), Softmax())

2. Operator overloading
-----------------------

The ``Model.define_operators()`` classmethod allows you to bind arbitrary
binary functions to Python operators, for use in any ``Model`` instance. The
method can (and should) be used as a context-manager, so that the overloading
is limited to the immediate block. This allows concise and expressive model
definition:

.. code:: python

    with Model.define_operators({'>>': chain}):
        model = ReLu(512) >> ReLu(512) >> Softmax()

The overloading is cleaned up at the end of the block. Only a few functions are
currently implemented. The three most useful are:

* ``chain(model1, model2)``: Compose two models ``f(x)`` and ``g(x)`` into a single model computing ``g(f(x))``.

* ``clone(model1, int)``: Create ``n`` copies of a model, each with distinct weights, and chain them together.

* ``concatenate(model1, model2)``: Given two models with output dimensions ``(n,)`` and ``(m,)``, construct a model with output dimensions ``(m+n,)``.

Putting these things together, here's the sort of tagging model that Thinc is
designed to make easy.

.. code:: python

    with Model.define_operators({'>>': chain, '**': clone, '|': concatenate}):
        model = (
            add_eol_markers('EOL')
            >> flatten
            >> memoize(
                CharLSTM(char_width)
                | (normalize >> str2int >> Embed(word_width)))
            >> ExtractWindow(nW=2)
            >> BatchNorm(ReLu(huidden_width)) ** 3
            >> Softmax()
        ) 

Not all of these pieces are implemented yet, but hopefully this shows where
we're going. The ``memoize`` function will be particularly important: in any
batch of text, the common words will be very common. It's therefore important
to evaluate models such as the ``CharLSTM`` once per word type per minibatch,
rather than once per token.

3. Callback-based backpropagation
---------------------------------

Most neural network libraries use a computational graph abstraction. This takes
the execution away from you, so that gradients can be computed automatically.
Thinc follows a style more like the ``autograd`` library, but with larger
operations. Usage is as follows:

.. code:: python

    def explicit_sgd_update(X, y):
        sgd = lambda weights, gradient: weights - gradient * 0.001
        yh, finish_update = model.begin_update(X, drop=0.2)
        finish_update(y-yh, optimizer)

Separating the backpropagation into three parts like this has many advantages.
The interface to all models is completely uniform — there is no distinction
between the top-level model you use as a predictor and the internal models for
the layers. We also make concurrency simple, by making the ``begin_update()``
step a pure function, and separating the accumulation of the gradient from the
action of the optimizer.

4. Class annotations
--------------------

To keep the class hierarchy shallow, Thinc uses class decorators to reuse code
for layer definitions. Specifically, the following decorators are available:

* ``describe.attributes()``: Allows attributes to be specified by keyword argument. Used especially for dimensions and parameters. 

* ``describe.on_init()``: Allows callbacks to be specified, which will be called at the end of the ``__init__.py``.

* ``describe.on_data()``: Allows callbacks to be specified, which will be called on ``Model.begin_training()``.
