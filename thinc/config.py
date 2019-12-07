import configparser
from pathlib import Path


class Config(dict):
    def __init__(self, data=None):
        dict.__init__(self)
        if data is None:
            data = {}
        self.update(data)

    def interpret_config(self, config):
        for section, values in config.items():
            parts = section.split(".")
            node = self
            for part in parts:
                node = node.setdefault(part, {})
            for key, value in values.items():
                if is_float(value):
                    node[key] = config.getfloat(section, key)
                elif is_int(value):
                    node[key] = config.getint(section, key)
                elif is_bool(value):
                    node[key] = config.getboolean(section, key)
                else:
                    value = strip_quotes(config.get(section, key))
                    node[key] = value

    def from_bytes(self, byte_string):
        text = byte_string.decode("utf8")
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        config.read_string(text)
        for key in list(self.keys()):
            self.pop(key)
        self.interpret_config(config)
        return self

    def from_disk(self, path):
        with Path(path).open("rb", encoding="utf8") as file_:
            data = file_.read()
        return self.from_bytes(data)


def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def is_bool(value):
    return value.lower() in ["true", "false"]


def strip_quotes(value):
    if len(value) < 3:
        return value
    elif value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    elif value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    else:
        return value
