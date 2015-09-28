=============================================
thinc: Learn super-sparse multi-class models
=============================================

.. image:: https://travis-ci.org/honnibal/thinc.svg?branch=master
    :target: https://travis-ci.org/honnibal/thinc

thinc is a Cython library for learning models with millions of parameters and
dozens of classes.  It drives http://honnibal.github.io/spaCy , a pipeline of very efficient NLP components.
I've only used thinc from Cython; no real Python API is currently available.

Currently the only model implemented is the averaged perceptron, which is
surprisingly competitive for these problems.

Despite the recent enthusiasm for deep learning, linear models can still
perform very well, if the right feature engineering is applied.  The key is
adding good conjunction features --- e.g., "next_word=X && next_next_word=Y".
For this, I have a helper-class thinc.features.Extractor, which you pass a list
of templates, which then performs your feature extraction, given an array of
atomic context items.

License
-------

The MIT License (MIT)

Copyright (C) 2015 Matthew Honnibal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
