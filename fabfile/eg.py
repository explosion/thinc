from fabric.api import task, local, run, lcd, cd, env
from os.path import exists as file_exists
from fabtools.python import virtualenv
from os import path


PWD = path.join(path.dirname(__file__), '..')
VENV_DIR = path.join(PWD, '.env')


@task
def mnist():
    with virtualenv(VENV_DIR):
        with lcd(PWD):
            local('pip install -e .')
            local('pip install keras')
            print("Using Keras to get MNIST data")
            local('KERAS_BACKEND="theano" python examples/mnist.py')


@task
def basic_tagger():
    with virtualenv(VENV_DIR):
        with lcd(PWD):
            local('pip install -e .')
            local('mkdir data')
            install_ancora()
            local('python examples/basic_tagger.py')
