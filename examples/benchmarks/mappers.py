from thinc.api import remap_ids_v2
from thinc.api import premap_ids
import time
import random
import numpy as np
import cupy as cp


N_symbols = 200000
N_tokens = 50000
N_columns = 4
mapper = {}
numbers = list(range(N_symbols))
random.shuffle(numbers)
for v, k in enumerate(numbers):
    mapper[k] = v


def speed_test_no_column():
    start = time.process_time()
    remap = remap_ids_v2(mapper)
    premap = premap_ids(mapper)
    keys = np.random.randint(0, N_symbols, N_tokens)
    start = time.process_time()
    for i in range(100):
        remap(keys, False)
    remaptime = time.process_time() - start
    start = time.process_time()
    for i in range(100):
        premap(keys, False)
    premaptime = time.process_time() - start
    print("remap", remaptime)
    print("premap", premaptime)
    print("speedup", remaptime / premaptime)


def speed_test_column():
    start = time.process_time()
    remap = remap_ids_v2(mapper, column=3)
    premap = premap_ids(mapper, column=3)
    keys = np.random.randint(0, N_symbols, (N_tokens, N_columns))
    start = time.process_time()
    for i in range(100):
        remap(keys, False)
    remaptime = time.process_time() - start
    start = time.process_time()
    for i in range(100):
        premap(keys, False)
    premaptime = time.process_time() - start
    print("remap", remaptime)
    print("premap", premaptime)
    print("speedup", remaptime / premaptime)


def speed_test_cupy():
    start = time.process_time()
    remap = remap_ids_v2(mapper)
    premap = premap_ids(mapper)
    keys = cp.random.randint(0, N_symbols, N_tokens)
    start = time.process_time()
    for i in range(100):
        remap(keys, False)
    remaptime = time.process_time() - start
    start = time.process_time()
    for i in range(100):
        premap(keys, False)
    premaptime = time.process_time() - start
    print("remap", remaptime)
    print("premap", premaptime)
    print("speedup", remaptime / premaptime)


print("No columns")
speed_test_no_column()
print("Columns")
speed_test_column()
print("Cupy")
speed_test_cupy()
