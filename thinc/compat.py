# coding: utf8
from __future__ import unicode_literals
import sys


try:
    basestring = basestring
except NameError:
    basestring = str


try:
    from StringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO  # noqa: F401


# See: https://github.com/benjaminp/six/blob/master/six.py
is_python2 = sys.version_info[0] == 2
is_python3 = sys.version_info[0] == 3
is_python_pre_3_5 = is_python2 or (is_python3 and sys.version_info[1] < 5)

if is_python3:
    integer_types = (int,)
    string_types = (str,)
    from urllib.error import URLError, HTTPError
    from urllib.request import urlopen, urlretrieve
else:
    integer_types = (int, long)
    string_types = (basestring,)
    from urllib2 import URLError, HTTPError
    from urllib2 import urlopen
    from urllib import urlretrieve
