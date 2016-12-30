from fabric.api import task, local, run, lcd, cd, env
from os.path import exists as file_exists
from fabtools.python import virtualenv
from os import path


PWD = path.dirname(__file__)
VENV_DIR = path.join(PWD, '.env')


#def dev():
#    # Allow this to persist, since we aren't as rigorous about keeping state clean
#    if not file_exists('.denv'):
#        local('virtualenv .denv')
# 
#    with virtualenv(DEV_ENV_DIR):
#        local('pip install -r requirements.txt')

@task
def sdist():
    if file_exists('dist/'):
        local('rm -rf dist/')
    local('mkdir dist')
    with virtualenv(VENV_DIR):
        local('python setup.py sdist')

@task
def publish(version):
    with virtualenv(VENV_DIR):
        local('git push origin master')
        local('git tag -a %s' % version)
        local('git push origin %s' % version)
        local('python setup.py sdist')
        local('python setup.py register')
        local('twine upload dist/*.tar.gz')

@task
def env():
    if file_exists('.env'):
        local('rm -rf .env')
    local('virtualenv .env')
    local('pip install -r requirements.txt')

@task
def install():
    with virtualenv(VENV_DIR):
        local('pip install --upgrade setuptools')
        local('pip install dist/*.tar.gz')
        local('pip install pytest')

@task
def make():
    with virtualenv(VENV_DIR):
        with lcd(path.dirname(__file__)):
            local('python setup.py build_ext --inplace')

@task
def clean():
    with lcd(path.dirname(__file__)):
        local('python setup.py clean --all')

@task
def test():
    with virtualenv(VENV_DIR):
        with lcd(path.dirname(__file__)):
            local('python -m pytest -x thinc')
