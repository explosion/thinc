#!/usr/bin/env python
from __future__ import division, print_function
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


MAJOR      = 5
MINOR      = 0
MICRO      = 0
ISRELEASED = False
VERSION    = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


PACKAGES = [
    'thinc',
    'thinc.linear',
    'thinc.extra',
    'thinc.neural',
    'thinc.test',
    'thinc.linear.test',
    'thinc.extra.test',
    'thinc.neural.test'
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
    'thinc.neural.lvl0',
    'thinc.neural.nn',
]


if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[0:2] < (3, 4):
    raise RuntimeError('Python version 2.7 or >= 3.4 required.')


# By subclassing build_extensions we have the actual compiler that will be used which is really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
compile_options =  {'msvc'  : ['/Ox', '/EHsc'],
                    'other' : ['-O3', '-Wno-strict-prototypes', '-Wno-unused-function']}
link_options    =  {'msvc'  : [],
                    'other' : ['-lcblas']}

if sys.platform.startswith('darwin'):
    compile_options['other'].append('-mmacosx-version-min=10.8')
    compile_options['other'].append('-stdlib=libc++')
    link_options['other'].append('-lc++')


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


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of thinc.about messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists(os.path.join('thinc', 'about.py')):
        # must be a source distribution, use existing version file
        try:
            from thinc.about import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError('Unable to import git_revision. Try removing '
                              'thinc/about.py and the build directory '
                              'before building.')
    else:
        GIT_REVISION = 'Unknown'

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version(path):
    cnt = """# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    with open(path, 'w') as f:
        f.write(cnt % {'version': VERSION,
                       'full_version' : FULLVERSION,
                       'git_revision' : GIT_REVISION,
                       'isrelease': str(ISRELEASED)})


def generate_cython(root, source):
    print('Cythonizing sources', source)
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
        write_version(os.path.join(root, 'thinc', 'about.py'))

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
            generate_cython(root, 'tests')
            prepare_includes(root)

        setup(
            name='thinc',
            packages=PACKAGES,
            package_data={'': ['*.pyx', '*.pxd', '*.pxi']},
            description='Learn sparse linear models',
            author='Matthew Honnibal',
            author_email='matt@spacy.io',
            version=VERSION,
            url='https://github.com/honnibal/thinc',
            license='MIT',
            ext_modules=ext_modules,
            install_requires=['numpy', 'murmurhash>=0.26,<0.27', 'cymem>=1.30,<1.31', 'preshed>=0.46,<0.47'],
            cmdclass = {
                'build_ext': build_ext_subclass},
        )


if __name__ == '__main__':
    setup_package()
