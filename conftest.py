# NOTE: This is required for the MyPy test suite machinery to pick up
#       the test cases in `test_mypy.py`
import os

os.environ["MYPY_TEST_PREFIX"] = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "thinc", "tests",
)
pytest_plugins = ["mypy.test.data"]
