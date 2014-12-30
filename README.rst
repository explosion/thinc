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

thinc was written as part of the development of spaCy, which is dual-licensed:
GPL v3, or you can pay for a commercial license.  thinc is licensed in the same
way.  For a commercial license, contact honnibal@gmail.com

Copyright (C) 2014 Matthew Honnibal

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
