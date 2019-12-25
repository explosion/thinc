from __future__ import unicode_literals

import configparser
import json
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
                node[key] = json.loads(config.get(section, key))

    def from_str(self, text):
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        config.read_string(text)
        for key in list(self.keys()):
            self.pop(key)
        self.interpret_config(config)
        return self

    def from_bytes(self, byte_string):
        return self.from_str(byte_string.decode("utf8"))

    def from_disk(self, path):
        with Path(path).open("r", encoding="utf8") as file_:
            text = file_.read()
        return self.from_str(text)
