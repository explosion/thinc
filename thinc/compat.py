import sys

# See: https://github.com/benjaminp/six/blob/master/six.py
is_python2 = sys.version_info[0] == 2
is_python3 = sys.version_info[0] == 3
is_python_pre_3_5 = is_python2 or (is_python3 and sys.version_info[1] < 5)

integer_types = (int,)
string_types = (str,)
from urllib.error import URLError, HTTPError
from urllib.request import urlopen, urlretrieve
