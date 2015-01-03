#!/usr/bin/env python
import sys
import os
from os import path
from glob import glob
import subprocess


from setuptools import setup
from distutils.core import Extension
import shutil


pwd = os.path.dirname(__file__)
virtual_env = os.environ.get('VIRTUAL_ENV', '')

includes = ['.']
if virtual_env:
    includes += glob(os.path.join(virtual_env, 'include', 'site', '*'))
else:
    pass


libs = []


compile_args = []
link_args = []


exts = [
    Extension('thinc.learner', ['thinc/learner.cpp'],
              language="c++",
              include_dirs=includes,
              extra_compile_args=['-O3'] + compile_args,
              extra_link_args=['-O3'] + link_args),
    Extension('thinc.weights', ['thinc/weights.cpp'],
              language="c++",
              include_dirs=includes,
              extra_compile_args=['-O3'] + compile_args,
              extra_link_args=['-O3'] + link_args),
    Extension("thinc.features", ["thinc/features.cpp"],
              language="c++", include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension("thinc.search", ["thinc/search.cpp"], language="c++"),
    Extension("thinc.cache", ["thinc/cache.cpp"], include_dirs=includes, language="c++"),
    Extension("tests.c_test_search", ["tests/c_test_search.cpp"], include_dirs=includes, language="c++")
]


if sys.argv[1] == 'clean':
    print >> sys.stderr, "cleaning .c, .c++ and .so files matching sources"
    map(clean, exts)


setup(
    name='thinc',
    packages=['thinc'],
    version='1.64',
    description="Learn sparse linear models",
    author='Matthew Honnibal',
    author_email='honnibal@gmail.com',
    url="http://github.com/syllog1sm/thinc",
    package_data={"thinc": ["*.pyx", "*.pxd", "*.pxi"]},
    ext_modules=exts,
    install_requires=["murmurhash", "cymem", "preshed"]
)
