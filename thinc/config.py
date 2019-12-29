from typing import Union, Dict, Any, Optional, List, Tuple
from configparser import ConfigParser, ExtendedInterpolation
import json
import io
from pathlib import Path


class Config(dict):
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        dict.__init__(self)
        if data is None:
            data = {}
        self.update(data)

    def interpret_config(self, config: Union[Dict[str, Any], ConfigParser]):
        for section, values in config.items():
            parts = section.split(".")
            node = self
            for part in parts:
                node = node.setdefault(part, {})
            for key, value in values.items():
                node[key] = json.loads(config.get(section, key))

    def from_str(self, text: str) -> "Config":
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read_string(text)
        for key in list(self.keys()):
            self.pop(key)
        self.interpret_config(config)
        return self

    def to_str(self) -> str:
        flattened = ConfigParser(interpolation=ExtendedInterpolation())
        queue: List[Tuple[tuple, "Config"]] = [(tuple(), self)]
        for path, node in queue:
            for key, value in node.items():
                if hasattr(value, "items"):
                    queue.append((path + (key,), value))
                else:
                    assert path
                    section_name = ".".join(path)
                    if not flattened.has_section(section_name):
                        flattened.add_section(section_name)
                    flattened.set(section_name, key, json.dumps(value))
        string_io = io.StringIO()
        flattened.write(string_io)
        return string_io.getvalue()

    def to_bytes(self) -> bytes:
        return self.to_str().encode("utf8")

    def to_disk(self, path: Union[str, Path]):
        path = Path(path)
        with path.open("w", encoding="utf8") as file_:
            file_.write(self.to_str())

    def from_bytes(self, byte_string: bytes) -> "Config":
        return self.from_str(byte_string.decode("utf8"))

    def from_disk(self, path: Union[str, Path]) -> "Config":
        with Path(path).open("r", encoding="utf8") as file_:
            text = file_.read()
        return self.from_str(text)
