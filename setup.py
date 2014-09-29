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

compile_args = []
link_args = []

exts = [
    Extension('thinc.ml.learner', ['thinc/ml/learner.pyx'],
              language="c++",
              include_dirs=includes,
              extra_compile_args=['-O3'] + compile_args,
              extra_link_args=['-O3'] + link_args),
    Extension("thinc.features.extractor", ["thinc/features/extractor.pyx"],
              language="c++", include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    #Extension("thinc.context.example", ["thinc/context/example.pyx"], language="c++")
    #Extension("thinc.context.segment", ["thinc/context/segment.pyx", 'thinc/ext/MurmurHash2.cpp'], language="c++",
    #          include_dirs=includes),
    Extension("thinc.search.beam", ["thinc/search/beam.pyx"], language="c++")
]


if sys.argv[1] == 'clean':
    print >> sys.stderr, "cleaning .c, .c++ and .so files matching sources"
    map(clean, exts)

distutils.core.setup(
    name='thinc',
    packages=['thinc', 'thinc.ml', 'thinc.features'],
    version='1.0',
    author='Matthew Honnibal',
    author_email='honnibal@gmail.com',
    url="http://github.com/syllog1sm/thinc",
    package_data={"thinc":
        ["*.pyx", "*.pxd", "*.cpp", "*.c", "*/*.pxd", "*/*.pyx", "*/*.cpp", "*/*.c"],
    },
    cmdclass={'build_ext': Cython.Distutils.build_ext},
    ext_modules=exts,
)



