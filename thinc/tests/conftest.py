import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="include slow tests")


def pytest_runtest_setup(item):
    def getopt(opt):
        # When using 'pytest --pyargs thinc' to test an installed copy of
        # thinc, pytest skips running our pytest_addoption() hook. Later, when
        # we call getoption(), pytest raises an error, because it doesn't
        # recognize the option we're asking about. To avoid this, we need to
        # pass a default value. We default to False, i.e., we act like all the
        # options weren't given.
        return item.config.getoption(f"--{opt}", False)

    for opt in ["slow"]:
        if opt in item.keywords and not getopt(opt):
            pytest.skip(f"need --{opt} option to run")
