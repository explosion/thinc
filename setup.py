#!/usr/bin/env python
import subprocess
from setuptools import setup
from glob import glob

import sys
import os
from os import path
from os.path import splitext


from setuptools import Extension


def clean(ext):
    for src in ext.sources:
        if src.endswith('.c') or src.endswith('cpp'):
            so = src.rsplit('.', 1)[0] + '.so'
            html = src.rsplit('.', 1)[0] + '.html'
            if os.path.exists(so):
                os.unlink(so)
            if os.path.exists(html):
                os.unlink(html)


def name_to_path(mod_name, ext):
    return '%s.%s' % (mod_name.replace('.', '/'), ext)


def c_ext(mod_name, language, includes, compile_args):
    mod_path = name_to_path(mod_name, language)
    return Extension(mod_name, [mod_path], include_dirs=includes,
                     extra_compile_args=compile_args, extra_link_args=compile_args)


def cython_ext(mod_name, language, includes, compile_args):
    import Cython.Distutils
    import Cython.Build
    mod_path = mod_name.replace('.', '/') + '.pyx'
    if language == 'cpp':
        language = 'c++'
    ext = Extension(mod_name, [mod_path], language=language, include_dirs=includes,
                    extra_compile_args=compile_args)
    return Cython.Build.cythonize([ext])[0]


def run_setup(exts):
    setup(
        name='thinc',
        packages=['thinc'],
        version='1.75',
        description="Learn sparse linear models",
        author='Matthew Honnibal',
        author_email='honnibal@gmail.com',
        url="http://github.com/syllog1sm/thinc",
        package_data={"thinc": ["*.pyx", "*.pxd", "*.pxi"]},
        ext_modules=exts,
        install_requires=["murmurhash", "cymem", "preshed"],
        setup_requires=["headers_workaround"]
    )

    import headers_workaround

    headers_workaround.fix_venv_pypy_include()
    headers_workaround.install_headers('murmurhash')


def main(modules, use_cython):
    language = "cpp"
    ext_func = cython_ext if use_cython else c_ext
    includes = ['.', path.join(sys.prefix, 'include')]
    compile_args = ['-O3']
    exts = [ext_func(mn, language, includes, compile_args) for mn in modules]
    run_setup(exts)


MOD_NAMES = ['thinc.learner', 'thinc.weights', 'thinc.features',
             'thinc.search', 'thinc.cache', 'tests.c_test_search']


if __name__ == '__main__':
    use_cython = sys.argv[1] == 'build_ext'
    main(MOD_NAMES, use_cython)
