# coding: utf8
from __future__ import unicode_literals
from typing import Optional

from .neural._classes.hash_embed import HashEmbed  # noqa: F401
from .neural._classes.embed import Embed, SimpleEmbed  # noqa: F401
from .neural._classes.static_vectors import StaticVectors  # noqa: F401
from ._registry import registry


@registry.layers.register("HashEmbed.v1")
def make_HashEmbed(outputs: int, rows: int, column: int, seed: Optional[int]=None):
    return HashEmbed(outputs, rows, seed=seed, column=column)


@registry.layers.register("SimpleEmbed.v1")
def make_SimpleEmbed(outputs: int, rows: int, column: int):
    return SimpleEmbed(outputs, rows, column)


@registry.layers.register("EmbedAndProject.v1")
def make_EmbedAndProject(outputs: int, rows: int, column: int):
    return Embed(outputs, rows, column)


@registry.layers.register("StaticVectors.v1")
def make_StaticVectors(outputs: int, spacy_name: str, column: int):
    return StaticVectors(nO=outputs, lang=spacy_name, column=column)
