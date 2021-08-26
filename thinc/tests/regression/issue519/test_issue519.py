import subprocess
try:
    import importlib.resources as importlib_resources
except ImportError:
    import importlib_resources


import pytest
from thinc.api import chain, Linear


@pytest.mark.slow
def test_issue519():
    """
    Test ability of Thinc mypy plugin to handle variadic arguments.

    This test usually takes around 45 seconds (at least on my machine), so
    I've marked it as slow 🙂
    """
    # Determine the name of the parent module (which contains the test program)
    parent_module_name = __name__[:__name__.rfind(".")]

    # Load test program that calls a Thinc API with variadic arguments
    program_text = importlib_resources.read_text(parent_module_name, "program.py")

    # Ask Mypy to type-check the loaded program text
    subprocess.run(["mypy", "--command", program_text], check=True)
