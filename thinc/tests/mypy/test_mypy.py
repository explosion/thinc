import os
import re
from pathlib import Path

import pytest

from mypy import api as mypy_api

# This ensures mypy can find the test files, no matter where tests are run from:
os.chdir(Path(__file__).parent.parent.parent.parent)

# You can change the following variable to True during development to overwrite expected output with generated output
GENERATE = False

cases = [
    ("mypy-plugin.ini", "success_plugin.py", "success-plugin.txt"),
    ("mypy-plugin.ini", "fail_plugin.py", "fail-plugin.txt"),
    ("mypy-default.ini", "success_no_plugin.py", "success-no-plugin.txt"),
    ("mypy-default.ini", "fail_no_plugin.py", "fail-no-plugin.txt"),
]


@pytest.mark.parametrize("config_filename,python_filename,output_filename", cases)
def test_mypy_results(config_filename, python_filename, output_filename):
    # To debug output text files
    print(f"Running from: {os.getcwd()}")
    print(f"Current directory contains: {os.listdir()}")
    full_config_filename = f"thinc/tests/mypy/configs/{config_filename}"
    full_filename = f"thinc/tests/mypy/modules/{python_filename}"
    full_output_filename = f"thinc/tests/mypy/outputs/{output_filename}"

    expected_out = ""
    expected_err = ""
    expected_returncode = 0 if output_filename is None else 1
    if full_output_filename is not None:
        with open(full_output_filename, "r") as f:
            expected_out = f.read()

    # Specifying a different cache dir for each configuration dramatically speeds up subsequent execution
    # It also prevents cache-invalidation-related bugs in the tests
    cache_dir = f".mypy_cache/test-{config_filename[:-4]}"
    command = [
        full_filename,
        "--config-file",
        full_config_filename,
        "--cache-dir",
        cache_dir,
        "--show-error-codes",
    ]
    print(
        f"\nExecuting: mypy {' '.join(command)}"
    )  # makes it easier to debug as necessary
    actual_result = mypy_api.run(command)
    actual_out, actual_err, actual_returncode = actual_result
    # Need to strip filenames due to differences in formatting by OS
    actual_out = "\n".join(
        [".py:".join(line.split(".py:")[1:]) for line in actual_out.split("\n") if line]
    ).strip()
    actual_out = re.sub(r"\n\s*\n", r"\n", actual_out)

    if GENERATE and output_filename is not None:
        with open(full_output_filename, "w") as f:
            f.write(actual_out)
    else:
        assert actual_out == expected_out, actual_out

    assert actual_err == expected_err
    assert actual_returncode == expected_returncode


def test_generation_is_disabled():
    """
    Makes sure we don't accidentally leave generation on
    """
    assert not GENERATE
