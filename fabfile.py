from fabric.api import local, run, lcd, cd, env
from os.path import exists as file_exists
from fabtools.python import virtualenv
from os import path


PWD = path.dirname(__file__)
VENV_DIR = path.join(PWD, '.env')
DEV_ENV_DIR = path.join(PWD, '.denv')


def dev():
    # Allow this to persist, since we aren't as rigorous about keeping state clean
    if not file_exists('.denv'):
        local('virtualenv .denv')
 
    with virtualenv(DEV_ENV_DIR):
        local('pip install -r requirements.txt')



def sdist():
    if file_exists('dist/'):
        local('rm -rf dist/')
    local('mkdir dist')
    with virtualenv(VENV_DIR):
        local('python setup.py sdist')


def publish(version):
    with virtualenv(VENV_DIR):
        local('git push origin master')
        local('git tag -a %s' % version)
        local('git push origin %s' % version)
        local('python setup.py sdist')
        local('python setup.py register')
        local('twine upload dist/*.tar.gz')


def setup():
    if file_exists('.env'):
        local('rm -rf .env')
    local('virtualenv .env')


def install():
    with virtualenv(VENV_DIR):
        local('pip install --upgrade setuptools')
        local('pip install dist/*.tar.gz')
        local('pip install pytest')


def make():
    with virtualenv(VENV_DIR):
        with lcd(path.dirname(__file__)):
            local('python setup.py build')


def clean():
    with lcd(path.dirname(__file__)):
        local('python setup.py clean --all')


def test():
    with virtualenv(VENV_DIR):
        with lcd(path.dirname(__file__)):
            local('python -m pytest -x thinc')


def travis():
    local('open https://travis-ci.org/spacy-io/thinc')
