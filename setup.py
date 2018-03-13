#!/usr/bin/env python
from __future__ import print_function
import io
import os
import subprocess
import sys
import contextlib
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc
from distutils import ccompiler, msvccompiler

from setuptools import Extension, setup


PACKAGES = [
    'thinc',
    'thinc.tests',
    'thinc.tests.unit',
    'thinc.tests.integration',
    'thinc.tests.linear',
    'thinc.linear',
    'thinc.neural',
    'thinc.extra',
    'thinc.neural._classes',
    'thinc.extra._vendorized'
]


MOD_NAMES = [
    'thinc.linalg',
    'thinc.openblas',
    'thinc.structs',
    'thinc.typedefs',
    'thinc.linear.avgtron',
    'thinc.linear.features',
    'thinc.linear.serialize',
    'thinc.linear.sparse',
    'thinc.linear.linear',
    'thinc.neural.optimizers',
    'thinc.neural.ops',
    'thinc.neural.gpu_ops',
    'thinc.neural._aligned_alloc',
    'thinc.extra.eg',
    'thinc.extra.mb',
    'thinc.extra.search',
    'thinc.extra.cache',
]

compile_options =  {'msvc'  : ['/Ox', '/EHsc'],
                    'other' : {
                        'gcc': ['-O2', '-Wno-strict-prototypes', '-Wno-unused-function'],
                        'nvcc': ['-arch=sm_30', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]}}
link_options    =  {'msvc'  : [], 'other' : []}

def link_static_openblas(root):
    pxi_loc = os.path.join(root, 'thinc', 'compile_time_constants.pxi')
    with open(pxi_loc, 'r') as file_:
        pxi = file_.read()
    if 'THINC_CBLAS' in os.environ:
        lib_loc = os.environ['THINC_CBLAS']
        lib_path, lib_name = os.path.split(lib_loc)
        if lib_name.endswith('.so'):
            is_shared = True
            lib_name = lib_name[3:-3]
        else:
            is_shared = False
        print('Using BLAS:', lib_path, lib_name)
        compile_options['other']['gcc'].append('-L%s' % lib_path)
        link_options['other'].append('-L%s' % lib_path)
        if is_shared:
            compile_options['other']['gcc'].append('-l%s' % lib_name)
            link_options['other'].append('-l%s' % lib_name)
        else:
            compile_options['other']['gcc'].append('-l:%s' % lib_name)
            link_options['other'].append('-l:%s' % lib_name)
        pxi = pxi.replace('DEF USE_BLAS = False', 'DEF USE_BLAS = True')
    else:
        print('Not compiling BLAS')
        pxi = pxi.replace('DEF USE_BLAS = True', 'DEF USE_BLAS = False')
    with open(pxi_loc, 'w') as file_:
        file_.write(pxi)


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    if hasattr(self, 'compiler_so'):
        default_compiler_so = self.compiler_so
    else:
        # This was put in for Windows, but I'm running blind here...
        default_compiler_so = None
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu' and CUDA is not None:
            # use the cuda for .cu files
            if hasattr(self, 'set_executable'):
                # This was put in for Windows, but I'm running blind here...
                self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# By subclassing build_extensions we have the actual compiler that will be used which is really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
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
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


def generate_cython(root, source):
    print('Cythonizing sources')
    p = subprocess.call([sys.executable,
                         os.path.join(root, 'bin', 'cythonize.py'),
                         source], env=os.environ)
    if p != 0:
        raise RuntimeError('Running cythonize failed')


def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDA_HOME' in os.environ:
        home = os.environ['CUDA_HOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            print('Warning: The nvcc binary could not be located in your $PATH. '
                  'For GPU capability, either add it to your path, or set $CUDA_HOME')
            return None
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': os.path.join(home, 'include'),
                  'lib64': os.path.join(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            print('Warning: The CUDA %s path could not be located in %s' % (k, v))
            return None
    return cudaconfig

CUDA = locate_cuda()


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

    link_static_openblas(root)

    with chdir(root):
        with open(os.path.join(root, 'thinc', 'about.py')) as f:
            about = {}
            exec(f.read(), about)

        with io.open(os.path.join(root, 'README.rst'), encoding='utf8') as f:
            readme = f.read()

        include_dirs = [
            get_python_inc(plat_specific=True),
            os.path.join(root, 'include')]

        if (ccompiler.new_compiler().compiler_type == 'msvc'
            and msvccompiler.get_build_version() == 9):
            include_dirs.append(os.path.join(root, 'include', 'msvc9'))

        ext_modules = []
        for mod_name in MOD_NAMES:
            if mod_name.endswith('gpu_ops'):
                continue
            mod_path = mod_name.replace('.', '/') + '.cpp'
            ext_modules.append(
                Extension(mod_name, [mod_path],
                    language='c++', include_dirs=include_dirs
                ))
        if CUDA is None:
            pass
        else:
            with chdir(root):
                ext_modules.append(
                    Extension("thinc.neural.gpu_ops",
                        sources=["thinc/neural/gpu_ops.cpp", "include/_cuda_shim.cu"],
                        library_dirs=[CUDA['lib64']],
                        libraries=['cudart'],
                        language='c++',
                        runtime_library_dirs=[CUDA['lib64']],
                        # this syntax is specific to this build system
                        # we're only going to use certain compiler args with nvcc and not with gcc
                        # the implementation of this trick is in customize_compiler() below
                        extra_compile_args=['-arch=sm_30', '--ptxas-options=-v', '-c',
                                            '--compiler-options', "'-fPIC'"],
                        include_dirs = include_dirs + [CUDA['include']]))

        if not is_source_release(root):
            generate_cython(root, 'thinc')

        setup(
            name=about['__title__'],
            zip_safe=False,
            packages=PACKAGES,
            package_data={'': ['*.pyx', '*.pxd', '*.pxi', '*.cpp']},
            description=about['__summary__'],
            long_description=readme,
            author=about['__author__'],
            author_email=about['__email__'],
            version=about['__version__'],
            url=about['__uri__'],
            license=about['__license__'],
            ext_modules=ext_modules,
            install_requires=[
                'numpy>=1.7',
                'murmurhash>=0.28,<0.29',
                'cymem>=1.30,<1.32.0',
                'preshed>=1.0.0,<2.0.0',
                'hypothesis>=2,<3',
                'tqdm>=4.10.0,<5.0.0',
                'plac>=0.9,<1.0',
                'termcolor>=1.1.0,<1.2.0',
                'wrapt>=1.10.0,<1.11.0',
                'dill>=0.2.7,<0.3',
                'pathlib>=1.0.0,<2.0.0',
                'msgpack-python==0.5.4',
                'msgpack-numpy==0.4.1',
            ],
            classifiers=[
                'Development Status :: 5 - Production/Stable',
                'Environment :: Console',
                'Intended Audience :: Developers',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: MIT License',
                'Operating System :: POSIX :: Linux',
                'Operating System :: MacOS :: MacOS X',
                'Operating System :: Microsoft :: Windows',
                'Programming Language :: Cython',
                'Programming Language :: Python :: 2.6',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3.3',
                'Programming Language :: Python :: 3.4',
                'Programming Language :: Python :: 3.5',
                'Programming Language :: Python :: 3.6',
                'Topic :: Scientific/Engineering'],
            cmdclass = {
                'build_ext': build_ext_subclass},
        )


if __name__ == '__main__':
    setup_package()
