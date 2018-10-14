#!/usr/bin/env bash

set -e

# Insist repository is clean
git diff-index --quiet HEAD

git checkout v6.10.x
git pull origin v6.10.x
git push origin v6.10.x
version=$(grep "__version__ = " thinc/about.py)
version=${version/__version__ = }
version=${version/\'/}
version=${version/\'/}
git tag "v$version"
git push origin --tags
