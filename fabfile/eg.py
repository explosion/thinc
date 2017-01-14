# coding: utf-8
from __future__ import unicode_literals

from fabric.api import task, local, run, lcd, cd, env, shell_env
from os.path import exists as file_exists
from fabtools.python import virtualenv
from os import path


PWD = path.join(path.dirname(__file__), '..')
VENV_DIR = path.join(PWD, '.env')


@task
def mnist():
    with virtualenv(VENV_DIR), lcd(PWD), shell_env(PYTHONPATH=PWD):
        local('python examples/mnist_mlp.py')


@task
def basic_tagger():
    with virtualenv(VENV_DIR), lcd(PWD), shell_env(PYTHONPATH=PWD):
        local('python examples/basic_tagger.py')


@task
def cnn_tagger():
    with virtualenv(VENV_DIR), lcd(PWD), shell_env(PYTHONPATH=PWD):
        local('python examples/cnn_tagger.py')


@task
def spacy_tagger():
    with virtualenv(VENV_DIR), lcd(PWD), shell_env(PYTHONPATH=PWD):
        local('python examples/spacy_tagger.py')
