from fabric.api import local, run, lcd, cd, env
from fabtools.python import virtualenv
from os import path


PWD = path.dirname(__file__)
VENV_DIR = path.join(PWD, '.env')


def make():
    with virtualenv(VENV_DIR):
        with lcd(path.dirname(__file__)):
            local('python setup.py build_ext --inplace')

def clean():
    with lcd(path.dirname(__file__)):
        local('python setup.py clean --all')

def test():
    with virtualenv(VENV_DIR):
        with lcd(path.dirname(__file__)):
            local('py.test -x')
