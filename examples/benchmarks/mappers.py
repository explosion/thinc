from thinc.api import remap_ids_v2
from thinc.api import premap_ids
from thinc.api import chain, Embed, HashEmbed
from thinc.util import time_context
import time
import random
import numpy as np
import cupy as cp


N_symbols = 200000
N_tokens = 50000
N_batch = 500
N_columns = 4
N_dim = 300
mapper = {}
numbers = list(range(N_symbols))
random.shuffle(numbers)
for v, k in enumerate(numbers):
    mapper[k] = v


def speed_test_no_column():
    remap = remap_ids_v2(mapper)
    premap = premap_ids(mapper)
    keys = np.random.randint(0, N_symbols, N_tokens)
    with time_context() as elapsed:
        for i in range(100):
            remap(keys, False)
    remaptime = elapsed.elapsed
    with time_context() as elapsed:
        for i in range(100):
            premap(keys, False)
    premaptime = elapsed.elapsed
    print("remap", remaptime)
    print("premap", premaptime)
    print("speedup", remaptime / premaptime)


def speed_test_column():
    remap = remap_ids_v2(mapper, column=3)
    premap = premap_ids(mapper, column=3)
    keys = np.random.randint(0, N_symbols, (N_tokens, N_columns))
    with time_context() as elapsed:
        for i in range(100):
            remap(keys, False)
    remaptime = elapsed.elapsed
    with time_context() as elapsed:
        for i in range(100):
            premap(keys, False)
    premaptime = elapsed.elapsed
    print("remap", remaptime)
    print("premap", premaptime)
    print("speedup", remaptime / premaptime)


def speed_test_cupy():
    remap = remap_ids_v2(mapper)
    premap = premap_ids(mapper)
    keys = cp.random.randint(0, N_symbols, N_tokens)
    with time_context() as elapsed:
        for i in range(100):
            remap(keys, False)
    remaptime = elapsed.elapsed
    with time_context() as elapsed:
        for i in range(100):
            premap(keys, False)
    premaptime = elapsed.elapsed
    print("remap", remaptime)
    print("premap", premaptime)
    print("speedup", remaptime / premaptime)


def speed_test_with_embed():
    remap = chain(remap_ids_v2(mapper), Embed(N_dim, N_symbols))
    premap = chain(premap_ids(mapper), Embed(N_dim, N_symbols))
    remap.initialize()
    premap.initialize()
    keys = np.random.randint(0, N_symbols, N_tokens)
    with time_context() as elapsed:
        for i in range(100):
            remap(keys, False)
    remaptime = elapsed.elapsed
    with time_context() as elapsed:
        for i in range(100):
            premap(keys, False)
    premaptime = elapsed.elapsed
    print("remap", remaptime)
    print("premap", premaptime)
    print("speedup", remaptime / premaptime)


def speed_test_cupy_with_embed():
    remap = chain(remap_ids_v2(mapper), Embed(N_dim, N_symbols))
    premap = chain(premap_ids(mapper), Embed(N_dim, N_symbols))
    remap.initialize()
    premap.initialize()
    keys = cp.random.randint(0, N_symbols, N_tokens)
    with time_context() as elapsed:
        for i in range(100):
            remap(keys, False)
    remaptime = elapsed.elapsed
    with time_context() as elapsed:
        for i in range(100):
            premap(keys, False)
    premaptime = elapsed.elapsed
    print("remap", remaptime)
    print("premap", premaptime)
    print("speedup", remaptime / premaptime)


def speed_test_hashembed():
    embed = HashEmbed(N_dim, N_symbols)
    embed.initialize()
    keys = np.random.randint(0, N_symbols, N_tokens)
    with time_context() as elapsed:
        for i in range(100):
            embed(keys, False)
    print(elapsed.elapsed)


print("No columns")
speed_test_no_column()
print("Columns")
speed_test_column()
print("Cupy")
speed_test_cupy()
print("With Embed")
speed_test_with_embed()
print("Cupy With Embed")
speed_test_cupy_with_embed()
print("HashEmbed speed")
speed_test_hashembed()
