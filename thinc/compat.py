try:
    basestring = basestring
except NameError:
    basestring = str

try:
    from StringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
