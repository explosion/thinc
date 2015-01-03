from fabric.api import local, run, lcd, cd, env
from os.path import exists as file_exists
from fabtools.python import virtualenv
from os import path


PWD = path.dirname(__file__)
VENV_DIR = path.join(PWD, '.env')


def sdist():
    local('rm dist/*')
    with virtualenv(VENV_DIR):
        local('python setup.py sdist')


def install():
    if file_exists('.env'):
        local('rm -rf .env')
    local('virtualenv .env')
    with virtualenv(VENV_DIR):
        local('pip install --upgrade setuptools')
        local('pip install dist/*.tar.gz')
        local('pip install pytest')


def make():
    with virtualenv(VENV_DIR):
        with lcd(path.dirname(__file__)):
            local('python dev_setup.py build_ext --inplace')

def clean():
    with lcd(path.dirname(__file__)):
        local('python dev_setup.py clean --all')

def test():
    with virtualenv(VENV_DIR):
        with lcd(path.dirname(__file__)):
            local('py.test -x')
