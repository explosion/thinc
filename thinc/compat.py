# coding: utf8
from __future__ import unicode_literals


try:
    basestring = basestring
except NameError:
    basestring = str

try:
    from StringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO  # noqa: F401
