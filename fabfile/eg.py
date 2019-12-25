from fabric.api import task, local, lcd, shell_env
from fabtools.python import virtualenv

from _util import PWD, VENV_DIR


@task
def mnist():
    with virtualenv(VENV_DIR), lcd(PWD), shell_env(PYTHONPATH=PWD):
        local("python examples/mnist_mlp.py")


@task
def basic_tagger():
    with virtualenv(VENV_DIR), lcd(PWD), shell_env(PYTHONPATH=PWD):
        local("python examples/basic_tagger.py")


@task
def cnn_tagger():
    with virtualenv(VENV_DIR), lcd(PWD), shell_env(PYTHONPATH=PWD):
        local("python examples/cnn_tagger.py")


@task
def quora():
    with virtualenv(VENV_DIR), lcd(PWD), shell_env(PYTHONPATH=PWD):
        local("pip install spacy")
        local("python -m spacy.en.download")
        local("python examples/quora_similarity.py")


@task
def spacy_tagger():
    with virtualenv(VENV_DIR), lcd(PWD), shell_env(PYTHONPATH=PWD):
        local("python examples/spacy_tagger.py")
