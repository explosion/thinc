#!/usr/bin/env python
import Cython.Distutils
from distutils.extension import Extension
import distutils.core

import sys
import os
from os.path import join as pjoin


def clean(ext):
    for pyx in ext.sources:
        if pyx.endswith('.pyx'):
            c = pyx[:-4] + '.c'
            cpp = pyx[:-4] + '.cpp'
            so = pyx[:-4] + '.so'
            if os.path.exists(so):
                os.unlink(so)
            if os.path.exists(c):
                os.unlink(c)
            elif os.path.exists(cpp):
                os.unlink(cpp)


pwd = os.path.dirname(__file__)
virtual_env = os.environ.get('VIRTUAL_ENV', '')

includes = [os.path.join(virtual_env, 'include'),
            os.path.join(pwd, 'thinc/include'),
            os.path.join(pwd, 'thinc/ext'),
            os.path.join(pwd, 'thinc/ext/include')]
libs = [os.path.join(pwd, 'thinc/ext')]

compile_args = ['-fopenmp']
link_args = ['-fopenmp']

exts = [
    Extension('thinc.ext.murmurhash', ["thinc/ext/murmurhash.pyx",
              "thinc/ext/MurmurHash2.cpp", "thinc/ext/MurmurHash3.cpp"],
              language="c++", include_dirs=includes,
              extra_compile_args=compile_args, extra_link_args=link_args),
    Extension('thinc.ext.sparsehash', ["thinc/ext/sparsehash.pyx"], language="c++",
              include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension('thinc.ml.learner', ['thinc/ml/learner.pyx'], language="c++",
              include_dirs=includes,
              extra_compile_args=['-O3'] + compile_args,
              extra_link_args=['-O3'] + link_args),
    Extension("thinc.features.extractor",
              ["thinc/features/extractor.pyx", "thinc/ext/MurmurHash2.cpp",
              "thinc/ext/MurmurHash3.cpp"], language="c++", include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
]


if sys.argv[1] == 'clean':
    print >> sys.stderr, "cleaning .c, .c++ and .so files matching sources"
    map(clean, exts)

distutils.core.setup(
    name='Sparse linear models with Cython',
    packages=['thinc'],
    author='Matthew Honnibal',
    author_email='honnibal@gmail.com',
    version='1.0',
    cmdclass={'build_ext': Cython.Distutils.build_ext},
    ext_modules=exts,
)



