#!/usr/bin/env python
import Cython.Distutils
from distutils.extension import Extension
import distutils.core

import sys
import os
from os.path import join as pjoin
from glob import glob


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

includes = []
if virtual_env:
    includes += glob(os.path.join(virtual_env, 'include', 'site', '*'))
else:
    # If you're not using virtualenv, ensure MurmurHash3.h is findable here.
    import murmurhash
    includes.append(os.path.dirname(murmurhash.__file__))


libs = []

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
    Extension("thinc.search.beam", ["thinc/search/beam.pyx"], language="c++")
]


if sys.argv[1] == 'clean':
    print >> sys.stderr, "cleaning .c, .c++ and .so files matching sources"
    map(clean, exts)

distutils.core.setup(
    name='thinc',
    packages=['thinc', 'thinc.ml', 'thinc.features', 'thinc.search'],
    version='1.1',
    author='Matthew Honnibal',
    author_email='honnibal@gmail.com',
    url="http://github.com/syllog1sm/thinc",
    package_data={"thinc": ["__init__.pxd", "features/*.pxd", "ml/*.pxd",
                            "search/*.pxd"]
    },
    cmdclass={'build_ext': Cython.Distutils.build_ext},
    ext_modules=exts,
)



