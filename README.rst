Thinc: spaCy's Machine Learning library for NLP in Python
*********************************************************

Thinc is the machine learning library powering `spaCy <https://spacy.io>`_. spaCy currently uses sparse linear models with large numbers of features. For instance, the English dependency parsing model has over 9m features. Thinc's linear model is implemented in Cython to support these large workloads. It also features flexible and efficient beam search functionality.

For `spaCy v2.0 <https://github.com/explosion/spaCy/projects/3>`_, Thinc will be rewritten for the new deep learning models, with an emphasis on ease of installation, CPU performance and hierarchical, variable-length inputs.

.. image:: https://travis-ci.org/explosion/thinc.svg?branch=master
    :target: https://travis-ci.org/explosion/thinc
    :alt: Build Status

.. image:: https://img.shields.io/github/release/explosion/thinc.svg
    :target: https://github.com/explosion/thinc/releases   
    :alt: Current Release Version

.. image:: https://img.shields.io/pypi/v/thinc.svg   
    :target: https://pypi.python.org/pypi/thinc
    :alt: pypi Version
