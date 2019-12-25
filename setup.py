#!/usr/bin/env python
import os
import subprocess
import sys
import contextlib
import distutils.util
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc
from distutils import ccompiler, msvccompiler
from setuptools import Extension, setup
from pathlib import Path


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
    script = root / "bin" / "cythonize.py"
    p = subprocess.call([sys.executable, str(script), source], env=os.environ)
    if p != 0:
        raise RuntimeError("Running cythonize failed")


def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir_path in path.parts:
        bin_path = dir_path / name
        if bin_path.exists():
            return bin_path.resolve()
    return None


def clean(path):
    for name in MOD_NAMES:
        name = name.replace(".", "/")
        for ext in ["so", "html", "cpp", "c"]:
            file_path = path / f"{name}.{ext}"
            if file_path.exists():
                file_path.unlink()


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
    root = Path(__file__).parent

    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        return clean(root)

    with chdir(root):
        with (root / "thinc" / "about.py").open("r") as f:
            about = {}
            exec(f.read(), about)

        include_dirs = [get_python_inc(plat_specific=True), str(root / "include")]

        if (
            ccompiler.new_compiler().compiler_type == "msvc"
            and msvccompiler.get_build_version() == 9
        ):
            include_dirs.append(str(root / "include" / "msvc9"))

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

        if not (root / "PKG-INFO").exists():  # not source release
            generate_cython(root, "thinc")

        setup(
            name="thinc",
            packages=PACKAGES,
            version=about["__version__"],
            ext_modules=ext_modules,
            cmdclass={"build_ext": build_ext_subclass},
        )


setup_package()
