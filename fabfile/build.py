# coding: utf-8
from __future__ import unicode_literals, print_function

from fabric.api import task, local, lcd, env, settings, prefix
from fabtools.python import virtualenv
from os import path, environ


PWD = path.join(path.dirname(__file__), '..')
ENV = environ['VENV_DIR'] if 'VENV_DIR' in environ else '.env'
VENV_DIR = path.join(PWD, ENV)


@task
def env(lang='python2.7'):
    if path.exists(VENV_DIR):
        local('rm -rf {env}'.format(env=VENV_DIR))
    local('pip install virtualenv')
    local('python -m virtualenv -p {lang} {env}'.format(lang=lang, env=VENV_DIR))


@task
def install():
    with virtualenv(VENV_DIR):
        local('pip install --upgrade setuptools')
        local('pip install dist/*.tar.gz')
        local('pip install pytest')


@task
def make():
    with virtualenv(VENV_DIR):
        with lcd(PWD):
            local('pip install cython')
            local('pip install murmurhash')
            local('pip install -r requirements.txt')
            local('python setup.py build_ext --inplace')

@task
def sdist():
    with virtualenv(VENV_DIR):
        with lcd(PWD):
            local('python setup.py sdist')

@task
def clean():
    with lcd(PWD):
        local('python setup.py clean --all')


@task
def test():
    with virtualenv(VENV_DIR):
        with lcd(PWD):
            local('py.test -x spacy/tests')
