# coding: utf-8
from __future__ import unicode_literals, print_function

from fabric.api import task, local, run, lcd, cd, env, shell_env
from fabtools.python import virtualenv

from _util import PWD, VENV_DIR


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
