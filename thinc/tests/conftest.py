from typing import Any, Optional
import pytest
from .mypy_fixture import DataSuite, DataSuiteCollector


# This function name is special to pytest.  See
# https://docs.pytest.org/en/latest/reference.html#initialization-hooks
def pytest_addoption(parser: Any) -> None:
    parser.addoption("--slow", action="store_true", help="include slow tests")
    group = parser.getgroup("mypy")
    group.addoption(
        "--update-data",
        action="store_true",
        default=False,
        help="Update test data to reflect actual output"
        " (supported only for certain tests)",
    )
    group.addoption(
        "--save-failures-to",
        default=None,
        help="Copy the temp directories from failing tests to a target directory",
    )
    group.addoption(
        "--mypy-verbose",
        action="count",
        help="Set the verbose flag when creating mypy Options",
    )
    group.addoption(
        "--mypyc-showc",
        action="store_true",
        default=False,
        help="Display C code on mypyc test failures",
    )


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
        if opt in item.keywords and not getopt(f"--{opt}"):
            pytest.skip(f"need --{opt} option to run")


# This function name is special to pytest.  See
# http://doc.pytest.org/en/latest/writing_plugins.html#collection-hooks
def pytest_pycollect_makeitem(collector: Any, name: str, obj: object) -> Optional[Any]:
    """Called by pytest on each object in modules configured in conftest.py files.

    collector is pytest.Collector, returns Optional[pytest.Class]
    """
    if isinstance(obj, type):
        # Only classes derived from DataSuite contain test cases, not the DataSuite class itself
        if issubclass(obj, DataSuite) and obj is not DataSuite:
            # Non-None result means this obj is a test case.
            # The collect method of the returned DataSuiteCollector instance will be called later,
            # with self.obj being obj.
            return DataSuiteCollector(name, parent=collector)
    return None
