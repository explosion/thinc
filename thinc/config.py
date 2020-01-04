from typing import Union, Dict, Any, Optional, List, Tuple
from configparser import ConfigParser, ExtendedInterpolation
import io
from pathlib import Path
import srsly


class Config(dict):
    """This class holds the model and training configuration and can load and
    save the TOML-style configuration format from/to a string, file or bytes.
    The Config class is a subclass of dict and uses Python's ConfigParser
    under the hood.
    """

    def __init__(
        self, data: Optional[Union[Dict[str, Any], "ConfigParser", "Config"]] = None
    ) -> None:
        """Initialize a new Config object with optional data."""
        dict.__init__(self)
        if data is None:
            data = {}
        self.update(data)

    def interpret_config(self, config: Union[Dict[str, Any], "ConfigParser"]):
        """Interpret a config, parse nested sections and parse the values
        as JSON. Mostly used internally and modifies the config in place.
        """
        for section, values in config.items():
            if section == "DEFAULT":
                # Skip [DEFAULT] section for now since it causes validation
                # errors and we don't want to use it
                continue
            parts = section.split(".")
            node = self
            for part in parts:
                node = node.setdefault(part, {})
            for key, value in values.items():
                node[key] = srsly.json_loads(config.get(section, key))

    def from_str(self, text: str) -> "Config":
        "Load the config from a string."
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read_string(text)
        for key in list(self.keys()):
            self.pop(key)
        self.interpret_config(config)
        return self

    def to_str(self) -> str:
        """Write the config to a string."""
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
                    flattened.set(section_name, key, srsly.json_dumps(value))
        string_io = io.StringIO()
        flattened.write(string_io)
        return string_io.getvalue().strip()

    def to_bytes(self) -> bytes:
        """Serialize the config to a byte string."""
        return self.to_str().encode("utf8")

    def from_bytes(self, bytes_data: bytes) -> "Config":
        """Load the config from a byte string."""
        return self.from_str(bytes_data.decode("utf8"))

    def to_disk(self, path: Union[str, Path]):
        """Serialize the config to a file."""
        path = Path(path)
        with path.open("w", encoding="utf8") as file_:
            file_.write(self.to_str())

    def from_disk(self, path: Union[str, Path]) -> "Config":
        """Load config from a file."""
        with Path(path).open("r", encoding="utf8") as file_:
            text = file_.read()
        return self.from_str(text)
