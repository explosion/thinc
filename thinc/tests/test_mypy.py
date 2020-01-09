"""Mypy style test cases for Thinc."""

import os
import os.path
import sys
import re

import pytest  # type: ignore  # no pytest in typeshed
from .mypy_fixture import (
    DataDrivenTestCase,
    DataSuite,
    assert_string_arrays_equal,
    test_temp_dir,
)
from mypy import api

file_dir = os.path.dirname(os.path.realpath(__file__))
# Locations of test data files such as test case descriptions (.test).
test_data_prefix = os.path.join(file_dir, "test-data", "unit")


class ThincMyPyTests(DataSuite):
    files = ["thinc.test", "basic-types.test"]
    data_prefix = os.path.join(file_dir, "mypy")

    def run_case(self, testcase: DataDrivenTestCase) -> None:
        assert testcase.input is not None, "test was not properly set up"
        assert testcase.old_cwd is not None, "test was not properly set up"
        mypy_cmdline = [
            "--show-traceback",
            "--no-error-summary",
            # f"--config-file={inipath}/test_mypy.ini",
        ]
        version = sys.version_info[:2]
        mypy_cmdline.append(f"--python-version={'.'.join(map(str, version))}")

        program_text = "\n".join(testcase.input)
        flags = re.search("# flags: (.*)$", program_text, flags=re.MULTILINE)
        if flags:
            flag_list = flags.group(1).split()
            mypy_cmdline.extend(flag_list)

        # Write the program to a file.
        program_path = os.path.join(test_temp_dir, "main.py")
        mypy_cmdline.append(program_path)
        with open(program_path, "w") as file:
            for s in testcase.input:
                file.write("{}\n".format(s))
        output = []
        # Type check the program.
        out, err, returncode = api.run(mypy_cmdline)
        # split lines, remove newlines, and remove directory of test case
        for line in (out + err).splitlines():
            if line.startswith(test_temp_dir + os.sep):
                output.append(
                    line[len(test_temp_dir + os.sep) :]
                    .rstrip("\r\n")
                    .replace(".py", "")
                )
            else:
                output.append(line.rstrip("\r\n"))
        # Remove temp file.
        os.remove(program_path)
        assert testcase.output is not None, "test was not properly set up"
        assert_string_arrays_equal(
            testcase.output,
            output,
            "Invalid output ({}, line {})".format(testcase.file, testcase.line),
        )
