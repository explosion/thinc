# coding: utf-8
from __future__ import unicode_literals

from fabric.api import task, local, run, lcd, cd, env
from os.path import exists as file_exists
from fabtools.python import virtualenv
from os import path

from . import eg


PWD = path.join(path.dirname(__file__), '..')
VENV_DIR = path.join(PWD, '.env')



@task
def env():
    if file_exists('.env'):
        local('rm -rf .env')
    local('virtualenv .env')
    with virtualenv(VENV_DIR):
        local('pip install --upgrade setuptools')
        local('pip install -r requirements.txt')
        local('pip install pytest')

@task
def make():
    with virtualenv(VENV_DIR):
        with lcd(PWD):
            local('python setup.py build_ext --inplace')

@task
def clean():
    with lcd(PWD):
        local('python setup.py clean --all')

@task
def test():
    with virtualenv(VENV_DIR):
        with lcd(PWD):
            local('python -m pytest -x thinc')
