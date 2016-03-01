#!/usr/bin/env python
from __future__ import print_function
import os
import shutil
import subprocess
import sys
import contextlib
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc

try:
    from setuptools import Extension, setup
except ImportError:
    from distutils.core import Extension, setup


PACKAGES = [
    'thinc',
    'thinc.linear',
    'thinc.extra',
    'thinc.neural',
    'thinc.linear.tests',
    'thinc.extra.tests',
    'thinc.neural.tests'
]


MOD_NAMES = [
    'thinc.linalg',
    'thinc.structs',
    'thinc.typedefs',
    'thinc.linear.avgtron',
    'thinc.linear.features',
    'thinc.linear.serialize',
    'thinc.linear.sparse',
    'thinc.extra.eg',
    'thinc.extra.search',
    'thinc.extra.cache',
    'thinc.neural.nn',
    'thinc.neural.solve',
    'thinc.neural.forward',
    'thinc.neural.backward',
    'thinc.neural.tests._funcs_shim',
    'thinc.neural.tests._backprop_shim',
]


# By subclassing build_extensions we have the actual compiler that will be used which is really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
compile_options =  {'msvc'  : ['/Ox', '/EHsc'],
                    'other' : ['-O3', '-Wno-strict-prototypes', '-Wno-unused-function']}
link_options    =  {'msvc'  : [],
                    'other' : []}

class build_ext_options:
    def build_options(self):
        for e in self.extensions:
            e.extra_compile_args = compile_options.get(
                self.compiler.compiler_type, compile_options['other'])
        for e in self.extensions:
            e.extra_link_args = link_options.get(
                self.compiler.compiler_type, link_options['other'])


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


def generate_cython(root, source):
    print('Cythonizing sources')
    p = subprocess.call([sys.executable,
                         os.path.join(root, 'bin', 'cythonize.py'),
                         source])
    if p != 0:
        raise RuntimeError('Running cythonize failed')


def import_include(module_name):
    try:
        return __import__(module_name, globals(), locals(), [], 0)
    except ImportError:
        raise ImportError('Unable to import %s. Create a virtual environment '
                          'and install all dependencies from requirements.txt, '
                          'e.g., run "pip install -r requirements.txt".' % module_name)


def copy_include(src, dst, path):
    assert os.path.isdir(src)
    assert os.path.isdir(dst)
    if os.path.exists(os.path.join(dst, path)):
        shutil.rmtree(os.path.join(dst, path))
    shutil.copytree(
        os.path.join(src, path),
        os.path.join(dst, path))


def prepare_includes(path):
    include_dir = os.path.join(path, 'include')
    if not os.path.exists(include_dir):
        os.mkdir(include_dir)

    numpy = import_include('numpy')
    copy_include(numpy.get_include(), include_dir, 'numpy')

    murmurhash = import_include('murmurhash')
    copy_include(murmurhash.get_include(), include_dir, 'murmurhash')


def is_source_release(path):
    return os.path.exists(os.path.join(path, 'PKG-INFO'))


def clean(path):
    for name in MOD_NAMES:
        name = name.replace('.', '/')
        for ext in ['.so', '.html', '.cpp', '.c']:
            file_path = os.path.join(path, name + ext)
            if os.path.exists(file_path):
                os.unlink(file_path)


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        return clean(root)

    with chdir(root):
        about = {}
        with open(os.path.join(root, 'thinc', 'about.py')) as f:
            exec(f.read(), about)

        include_dirs = [
            get_python_inc(plat_specific=True),
            os.path.join(root, 'include')]

        ext_modules = []
        for mod_name in MOD_NAMES:
            mod_path = mod_name.replace('.', '/') + '.cpp'
            ext_modules.append(
                Extension(mod_name, [mod_path],
                    language='c++', include_dirs=include_dirs))

        if not is_source_release(root):
            generate_cython(root, 'thinc')
            prepare_includes(root)

        setup(
            name=about['__name__'],
            packages=PACKAGES,
            package_data={'': ['*.pyx', '*.pxd', '*.pxi']},
            description=about['__summary__'],
            author=about['__author__'],
            author_email=about['__email__'],
            version=about['__version__'],
            url=about['__uri__'],
            license=about['__license__'],
            ext_modules=ext_modules,
            install_requires=['numpy', 'murmurhash>=0.26,<0.27', 'cymem>=1.30,<1.32', 'preshed>=0.46,<0.47'],
            cmdclass = {
                'build_ext': build_ext_subclass},
        )


if __name__ == '__main__':
    setup_package()
