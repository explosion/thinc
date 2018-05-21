Thinc: Practical Machine Learning for NLP in Python
***************************************************

**Thinc** is the machine learning library powering `spaCy <https://spacy.io>`_.
It features a battle-tested linear model designed for large sparse learning
problems, and a flexible neural network model under development for
`spaCy v2.0 <https://spacy.io/usage/v2>`_.

Thinc is a practical toolkit for implementing models that follow the
`"Embed, encode, attend, predict" <https://explosion.ai/blog/deep-learning-formula-nlp>`_
architecture. It's designed to be easy to install, efficient for CPU usage and
optimised for NLP and deep learning with text – in particular, hierarchically
structured input and variable-length sequences.

🔮 **Version 6.11 out now!** `Read the release notes here. <https://github.com/explosion/thinc/releases/>`_

.. image:: https://img.shields.io/travis/explosion/thinc/master.svg?style=flat-square
    :target: https://travis-ci.org/explosion/thinc
    :alt: Build Status
    
.. image:: https://img.shields.io/appveyor/ci/explosion/thinc/master.svg?style=flat-square
    :target: https://ci.appveyor.com/project/explosion/thinc
    :alt: Appveyor Build Status

.. image:: https://img.shields.io/coveralls/explosion/thinc.svg?style=flat-square
    :target: https://coveralls.io/github/explosion/thinc
    :alt: Test Coverage

.. image:: https://img.shields.io/github/release/explosion/thinc.svg?style=flat-square
    :target: https://github.com/explosion/thinc/releases
    :alt: Current Release Version

.. image:: https://img.shields.io/pypi/v/thinc.svg?style=flat-square
    :target: https://pypi.python.org/pypi/thinc
    :alt: pypi Version

.. image:: https://anaconda.org/conda-forge/thinc/badges/version.svg
    :target: https://anaconda.org/conda-forge/thinc
    :alt: conda Version

.. image:: https://img.shields.io/badge/gitter-join%20chat%20%E2%86%92-7676d1.svg?style=flat-square
    :target: https://gitter.im/explosion/thinc
    :alt: Thinc on Gitter

.. image:: https://img.shields.io/twitter/follow/explosion_ai.svg?style=social&label=Follow
    :target: https://twitter.com/explosion_ai
    :alt: Follow us on Twitter

What's where (as of v6.9.0)
===========================

======================== ===
``thinc.v2v.Model``      Base class.
``thinc.v2v``            Layers transforming vectors to vectors.
``thinc.i2v``            Layers embedding IDs to vectors.
``thinc.t2v``            Layers pooling tensors to vectors.
``thinc.t2t``            Layers transforming tensors to tensors (e.g. CNN, LSTM).
``thinc.api``            Higher-order functions, for building networks. Will be renamed.
``thinc.extra``          Datasets and utilities.
``thinc.neural.ops``     Container classes for mathematical operations. Will be reorganized.
``thinc.linear.avgtron`` Legacy efficient Averaged Perceptron implementation.
======================== ===

Development status
==================

Thinc's deep learning functionality is still under active development: APIs are
unstable, and we're not yet ready to provide usage support. However, if you're
already quite familiar with neural networks, there's a lot here you might find
interesting. Thinc's conceptual model is quite different from TensorFlow's.
Thinc also implements some novel features, such as a small DSL for concisely
wiring up models, embedding tables that support pre-computation and the
hashing trick, dynamic batch sizes, a concatenation-based approach to
variable-length sequences, and support for model averaging for the
Adam solver (which performs very well).

No computational graph – just higher order functions
======================================================

The central problem for a neural network implementation is this: during the
forward pass, you compute results that will later be useful during the backward
pass. How do you keep track of this arbitrary state, while making sure that
layers can be cleanly composed?

Most libraries solve this problem by having you declare the forward
computations, which are then compiled into a graph somewhere behind the scenes.
Thinc doesn't have a "computational graph". Instead, we just use the stack,
because we put the state from the forward pass into callbacks.

All nodes in the network have a simple signature:

.. code:: none

    f(inputs) -> {outputs, f(d_outputs)->d_inputs}

To make this less abstract, here's a ReLu activation, following this signature:

.. code:: python

    def relu(inputs):
        mask = inputs > 0
        def backprop_relu(d_outputs, optimizer):
            return d_outputs * mask
        return inputs * mask, backprop_relu

When you call the ``relu`` function, you get back an output variable, and a
callback. This lets you calculate a gradient using the output, and then pass it
into the callback to perform the backward pass.

This signature makes it easy to build a complex network out of smaller pieces,
using arbitrary higher-order functions you can write yourself. To make this
clearer, we need a function for a weights layer. Usually this will be
implemented as a class — but let's continue using closures, to keep things
concise, and to keep the simplicity of the interface explicit.

The main complication for the weights layer is that we now have a side-effect
to manage: we would like to update the weights. There are a few ways to handle
this. In Thinc we currently pass a callable into the backward pass. (I'm not
convinced this is best.)

.. code:: python

    import numpy

    def create_linear_layer(n_out, n_in):
        W = numpy.zeros((n_out, n_in))
        b = numpy.zeros((n_out, 1))

        def forward(X):
            Y = W @ X + b
            def backward(dY, optimizer):
                dX = W.T @ dY
                dW = numpy.einsum('ik,jk->ij', dY, X)
                db = dY.sum(axis=0)

                optimizer(W, dW)
                optimizer(b, db)

                return dX
            return Y, backward
        return forward

If we call ``Wb = create_linear_layer(5, 4)``, the variable ``Wb`` will be the
``forward()`` function, implemented inside the body of ``create_linear_layer()``.
The `Wb` instance will have access to the ``W`` and ``b`` variable defined in its
outer scope. If we invoke ``create_linear_layer()`` again, we get a new instance,
with its own internal state.

The ``Wb`` instance and the ``relu`` function have exactly the same signature. This
makes it easy to write higher order functions to compose them. The most obvious
thing to do is chain them together:

.. code:: python

    def chain(*layers):
        def forward(X):
            backprops = []
            Y = X
            for layer in layers:
                Y, backprop = layer(Y)
                backprops.append(backprop)
            def backward(dY, optimizer):
                for backprop in reversed(backprops):
                    dY = backprop(dY, optimizer)
                return dY
            return Y, backward
        return forward

We could now chain our linear layer together with the ``relu`` activation, to
create a simple feed-forward network:

.. code:: python

    Wb1 = create_linear_layer(10, 5)
    Wb2 = create_linear_layer(3, 10)

    model = chain(Wb1, relu, Wb2)

    X = numpy.random.uniform(size=(5, 4))

    y, bp_y = model(X)

    dY = y - truth
    dX = bp_y(dY, optimizer)

This conceptual model makes Thinc very flexible. The trade-off is that Thinc is
less convenient and efficient at workloads that fit exactly into what
`Tensorflow <https://www.tensorflow.org/>`_ etc. are designed for. If your graph
really is static, and your inputs are homogenous in size and shape,
`Keras <https://keras.io/>`_ will likely be faster and simpler. But if you want
to pass normal Python objects through your network, or handle sequences and recursions
of arbitrary length or complexity, you might find Thinc's design a better fit for
your problem.

Quickstart
==========

Thinc should install cleanly with both `pip <http://pypi.python.org/pypi/thinc>`_ and
`conda <https://anaconda.org/conda-forge/thinc>`_, for **Pythons 2.7+ and 3.5+**, on
**Linux**, **macOS / OSX** and **Windows**.  Its only system dependency is a compiler
tool-chain (e.g. ``build-essential``) and the  Python development headers (e.g.
``python-dev``).

.. code:: bash

    pip install thinc

For GPU support, we're grateful to use the work of Chainer's cupy module, which provides a numpy-compatible interface for GPU arrays. However, installing Chainer when no GPU is available currently causes an error. We therefore do not list Chainer as an explicit dependency --- so building ``Thinc`` for GPU requires some extra steps:

.. code:: bash

    export CUDA_HOME=/usr/local/cuda-8.0 # Or wherever your CUDA is
    export PATH=$PATH:$CUDA_HOME/bin
    pip install chainer
    python -c "import cupy; assert cupy" # Check it installed
    pip install thinc
    python -c "import thinc.neural.gpu_ops" # Check the GPU ops were built

The rest of this section describes how to build Thinc from source. If you have
`Fabric <http://www.fabfile.org>`_ installed, you can use the shortcut:

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

Customizing the matrix multiplication backend
=============================================

Prior to v6.11, Thinc relied on numpy for matrix multiplications. When numpy is installed via wheel using pip (the default), numpy will usually be linked against a suboptimal matrix multiplication kernel. This made it difficult to ensure that Thinc was well optimized for the target machine.

To fix this, Thinc now provides its own matrix multiplications, by bundling the source code for OpenBLAS's sgemm kernel within the library. To change the default BLAS library, you can specify an environment variable, giving the location of the shared library you want to link against:

.. code:: bash
  
    THINC_BLAS=/opt/openblas/lib/libopenblas.so pip install thinc --no-cache-dir --no-binary
    export LD_LIBRARY_PATH=/opt/openblas/lib
    # On OSX:
    # export DYLD_LIBRARY_PATH=/opt/openblas/lib

If you want to link against the Intel MKL instead of OpenBLAS, the easiest way is to install Miniconda. For instance, if you installed miniconda to `/opt/miniconda', the command to install Thinc linked against MKL would be:

.. code:: bash
    
    THINC_BLAS=/opt/miniconda/numpy-mkl/lib/libmkl_rt.so pip install thinc --no-cache-dir --no-binary
    export LD_LIBRARY_PATH=/opt/miniconda/numpy-mkl/lib
    # On OSX:
    # export DYLD_LIBRARY_PATH=/opt/miniconda/numpy-mkl/lib

If the library file ends in a .a extension, it is linked statically; if it ends in .so, it's linked dynamically. Make sure you have the directory on your LD_LIBRARY_PATH at runtime if you use the dynamic linking.

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
    model = chain(ReLu(512, 748), ReLu(512, 784), Softmax(10))

    # Leave the dimensions unspecified, and you can't be wrong.
    model = chain(ReLu(512), ReLu(512), Softmax())

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

The overloading is cleaned up at the end of the block. A fairly arbitrary zoo
of functions are currently implemented. Some of the most useful:

* ``chain(model1, model2)``: Compose two models ``f(x)`` and ``g(x)`` into a single model computing ``g(f(x))``.

* ``clone(model1, int)``: Create ``n`` copies of a model, each with distinct weights, and chain them together.

* ``concatenate(model1, model2)``: Given two models with output dimensions ``(n,)`` and ``(m,)``, construct a model with output dimensions ``(m+n,)``.

* ``add(model1, model2)``: ``add(f(x), g(x)) = f(x)+g(x)``

* ``make_tuple(model1, model2)``: Construct tuples of the outputs of two models, at the batch level. The backward pass expects to receive a tuple of gradients, which are routed through the appropriate model, and summed.

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
            >> BatchNorm(ReLu(hidden_width)) ** 3
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
        finish_update(y-yh, sgd)

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

🛠 Changelog
============

=========== ============== ===========
Version     Date           Description
=========== ============== ===========
`v6.11.1`_  ``2018-05-20`` Support direct linkage to BLAS libraries
 v6.11.0    ``2018-03-16`` *n/a*
`v6.10.2`_  ``2017-12-06`` Efficiency improvements and bug fixes
`v6.10.1`_  ``2017-11-15`` Fix GPU install and minor memory leak
`v6.10.0`_  ``2017-10-28`` CPU efficiency improvements, refactoring
`v6.9.0`_   ``2017-10-03`` Reorganize layers, bug fix to Layer Normalization
`v6.8.2`_   ``2017-09-26`` Fix packaging of `gpu_ops`
`v6.8.1`_   ``2017-08-23`` Fix Windows support
`v6.8.0`_   ``2017-07-25`` SELU layer, attention, improved GPU/CPU compatibility
`v6.7.3`_   ``2017-06-05`` Fix convolution on GPU
`v6.7.2`_   ``2017-06-02`` Bug fixes to serialization
`v6.7.1`_   ``2017-06-02`` Improve serialization
`v6.7.0`_   ``2017-06-01`` Fixes to serialization, hash embeddings and flatten ops
`v6.6.0`_   ``2017-05-14`` Improved GPU usage and examples
 v6.5.2     ``2017-03-20`` *n/a*
`v6.5.1`_   ``2017-03-20`` Improved linear class and Windows fix
`v6.5.0`_   ``2017-03-11`` Supervised similarity, fancier embedding and improvements to linear model
 v6.4.0     ``2017-02-15`` *n/a*
`v6.3.0`_   ``2017-01-25`` Efficiency improvements, argument checking and error messaging
`v6.2.0`_   ``2017-01-15`` Improve API and introduce overloaded operators
`v6.1.3`_   ``2017-01-10`` More neural network functions and training continuation
 v6.1.3     ``2017-01-09`` *n/a*
 v6.1.2     ``2017-01-09`` *n/a*
 v6.1.1     ``2017-01-09`` *n/a*
 v6.1.0     ``2017-01-09`` *n/a*
`v6.0.0`_   ``2016-12-31`` Add ``thinc.neural`` for NLP-oriented deep learning
=========== ============== ===========

.. _v6.11.1: https://github.com/explosion/thinc/releases/tag/v6.11.1
.. _v6.10.2: https://github.com/explosion/thinc/releases/tag/v6.10.2
.. _v6.10.2: https://github.com/explosion/thinc/releases/tag/v6.10.2
.. _v6.10.1: https://github.com/explosion/thinc/releases/tag/v6.10.1
.. _v6.10.0: https://github.com/explosion/thinc/releases/tag/v6.10.0
.. _v6.9.0: https://github.com/explosion/thinc/releases/tag/v6.9.0
.. _v6.8.2: https://github.com/explosion/thinc/releases/tag/v6.8.2
.. _v6.8.1: https://github.com/explosion/thinc/releases/tag/v6.8.1
.. _v6.8.0: https://github.com/explosion/thinc/releases/tag/v6.8.0
.. _v6.7.3: https://github.com/explosion/thinc/releases/tag/v6.7.3
.. _v6.7.2: https://github.com/explosion/thinc/releases/tag/v6.7.2
.. _v6.7.1: https://github.com/explosion/thinc/releases/tag/v6.7.1
.. _v6.7.0: https://github.com/explosion/thinc/releases/tag/v6.7.0
.. _v6.6.0: https://github.com/explosion/thinc/releases/tag/v6.6.0
.. _v6.5.1: https://github.com/explosion/thinc/releases/tag/v6.5.1
.. _v6.5.0: https://github.com/explosion/thinc/releases/tag/v6.5.0
.. _v6.3.0: https://github.com/explosion/thinc/releases/tag/v6.3.0
.. _v6.2.0: https://github.com/explosion/thinc/releases/tag/v6.2.0
.. _v6.1.3: https://github.com/explosion/thinc/releases/tag/v6.1.3
.. _v6.0.0: https://github.com/explosion/thinc/releases/tag/v6.0.0
