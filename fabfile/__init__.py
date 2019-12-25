from fabric.api import task, local, lcd
from os.path import exists as file_exists
from fabtools.python import virtualenv

from . import eg  # noqa: F401
from _util import PWD, VENV_DIR


@task
def env():
    if file_exists(VENV_DIR):
        local("rm -rf {env}".format(env=VENV_DIR))
    local("python -m virtualenv {env}".format(env=VENV_DIR))
    with virtualenv(VENV_DIR):
        local("python -m pip install --upgrade setuptools")
        local("python -m pip install -r requirements.txt")
        local("python -m pip install pytest")


@task
def make():
    with virtualenv(VENV_DIR):
        with lcd(PWD):
            local("python setup.py build_ext --inplace")


@task
def clean():
    with lcd(PWD):
        local("python setup.py clean --all")


@task
def test():
    with virtualenv(VENV_DIR):
        with lcd(PWD):
            local("python -m pytest -x thinc")
