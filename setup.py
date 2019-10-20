#!/usr/bin/env python
from __future__ import print_function
import io
import os
import os.path
import subprocess
import sys
import contextlib
import distutils.util
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc
from distutils import ccompiler, msvccompiler
from distutils.ccompiler import new_compiler
import platform

from setuptools import Extension, setup


def is_new_osx():
    """Check whether we're on OSX >= 10.10"""
    name = distutils.util.get_platform()
    if sys.platform != "darwin":
        return False
    elif name.startswith("macosx-10"):
        minor_version = int(name.split("-")[1].split(".")[1])
        if minor_version >= 7:
            return True
        else:
            return False
    else:
        return False


PACKAGES = [
    "thinc",
    "thinc.tests",
    "thinc.tests.unit",
    "thinc.tests.integration",
    "thinc.tests.linear",
    "thinc.linear",
    "thinc.neural",
    "thinc.extra",
    "thinc.neural._classes",
    "thinc.extra._vendorized",
    "thinc.extra.wrapt",
]


MOD_NAMES = [
    "thinc.linalg",
    "thinc.structs",
    "thinc.typedefs",
    "thinc.linear.avgtron",
    "thinc.linear.features",
    "thinc.linear.serialize",
    "thinc.linear.sparse",
    "thinc.linear.linear",
    "thinc.neural.optimizers",
    "thinc.neural.ops",
    "thinc.neural.gpu_ops",
    "thinc.neural._aligned_alloc",
    #'thinc.neural._fast_maxout_cnn',
    "thinc.extra.eg",
    "thinc.extra.mb",
    "thinc.extra.search",
    "thinc.extra.cache",
]

COMPILE_OPTIONS = {
    "msvc": ["/Ox", "/EHsc"],
    "other": ["-O3", "-Wno-strict-prototypes", "-Wno-unused-function"],
}
LINK_OPTIONS = {"msvc": [], "other": []}


if is_new_osx():
    # On Mac, use libc++ because Apple deprecated use of
    # libstdc
    COMPILE_OPTIONS["other"].append("-stdlib=libc++")
    LINK_OPTIONS["other"].append("-lc++")
    # g++ (used by unix compiler on mac) links to libstdc++ as a default lib.
    # See: https://stackoverflow.com/questions/1653047/avoid-linking-to-libstdc
    LINK_OPTIONS["other"].append("-nodefaultlibs")

# By subclassing build_extensions we have the actual compiler that will be used
# which is really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_options:
    def build_options(self):
        src_dir = os.path.join(os.path.dirname(__file__), "thinc", "_files")
        if hasattr(self.compiler, "initialize"):
            self.compiler.initialize()
        self.compiler.platform = sys.platform[:6]
        for e in self.extensions:
            e.extra_compile_args = COMPILE_OPTIONS.get(
                self.compiler.compiler_type, COMPILE_OPTIONS["other"]
            )
            e.extra_link_args = LINK_OPTIONS.get(
                self.compiler.compiler_type, LINK_OPTIONS["other"]
            )


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


def generate_cython(root, source):
    print("Cythonizing sources")
    p = subprocess.call(
        [sys.executable, os.path.join(root, "bin", "cythonize.py"), source],
        env=os.environ,
    )
    if p != 0:
        raise RuntimeError("Running cythonize failed")


def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def is_source_release(path):
    return os.path.exists(os.path.join(path, "PKG-INFO"))


def clean(path):
    for name in MOD_NAMES:
        name = name.replace(".", "/")
        for ext in [".so", ".html", ".cpp", ".c"]:
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

    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        return clean(root)

    with chdir(root):
        with open(os.path.join(root, "thinc", "about.py")) as f:
            about = {}
            exec(f.read(), about)

        with io.open(os.path.join(root, "README.md"), encoding="utf8") as f:
            readme = f.read()

        include_dirs = [
            get_python_inc(plat_specific=True),
            os.path.join(root, "include"),
        ]

        if (
            ccompiler.new_compiler().compiler_type == "msvc"
            and msvccompiler.get_build_version() == 9
        ):
            include_dirs.append(os.path.join(root, "include", "msvc9"))

        ext_modules = []
        for mod_name in MOD_NAMES:
            mod_path = mod_name.replace(".", "/") + ".cpp"
            if mod_name.endswith("gpu_ops"):
                continue
            mod_path = mod_name.replace(".", "/") + ".cpp"
            ext_modules.append(
                Extension(
                    mod_name, [mod_path], language="c++", include_dirs=include_dirs
                )
            )
        ext_modules.append(
            Extension(
                "thinc.extra.wrapt._wrappers",
                ["thinc/extra/wrapt/_wrappers.c"],
                include_dirs=include_dirs,
            )
        )

        if not is_source_release(root):
            generate_cython(root, "thinc")

        setup(
            name="thinc",
            zip_safe=False,
            packages=PACKAGES,
            package_data={"": ["*.pyx", "*.pxd", "*.pxi", "*.cpp", "*.cu"]},
            description=about["__summary__"],
            long_description=readme,
            long_description_content_type="text/markdown",
            author=about["__author__"],
            author_email=about["__email__"],
            version=about["__version__"],
            url=about["__uri__"],
            license=about["__license__"],
            ext_modules=ext_modules,
            setup_requires=["numpy>=1.7.0"],
            install_requires=[
                # Explosion-provided dependencies
                "murmurhash>=0.28.0,<1.1.0",
                "cymem>=2.0.2,<2.1.0",
                "preshed>=1.0.1,<3.1.0",
                "blis>=0.4.0,<0.5.0",
                "wasabi>=0.0.9,<1.1.0",
                "srsly>=0.0.6,<1.1.0",
                # Third-party dependencies
                "numpy>=1.7.0",
                "plac>=0.9.6,<1.0.0",
                "tqdm>=4.10.0,<5.0.0",
                'pathlib==1.0.1; python_version < "3.4"',
            ],
            extras_require={
                "cuda": ["cupy>=5.0.0b4"],
                "cuda80": ["cupy-cuda80>=5.0.0b4"],
                "cuda90": ["cupy-cuda90>=5.0.0b4"],
                "cuda91": ["cupy-cuda91>=5.0.0b4"],
                "cuda92": ["cupy-cuda92>=5.0.0b4"],
                "cuda100": ["cupy-cuda100>=5.0.0b4"],
                "cuda110": ["cupy-cuda110>=5.0.0b4"],
            },
            classifiers=[
                "Development Status :: 5 - Production/Stable",
                "Environment :: Console",
                "Intended Audience :: Developers",
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Operating System :: POSIX :: Linux",
                "Operating System :: MacOS :: MacOS X",
                "Operating System :: Microsoft :: Windows",
                "Programming Language :: Cython",
                "Programming Language :: Python :: 2.6",
                "Programming Language :: Python :: 2.7",
                "Programming Language :: Python :: 3.3",
                "Programming Language :: Python :: 3.4",
                "Programming Language :: Python :: 3.5",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.8",
                "Topic :: Scientific/Engineering",
            ],
            cmdclass={"build_ext": build_ext_subclass},
        )


setup_package()
