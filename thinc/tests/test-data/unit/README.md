## thinc mypy tests

You'll notice the weird nested folders. It's because the underlying test machinery for mypy's suite is kind of hardcoded. If a path "test-data/init" does not exist as a subfolder of MYPY_TEST_PREFIX in the environment, everything will fail to run.
