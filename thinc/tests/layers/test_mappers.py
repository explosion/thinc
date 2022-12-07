from thinc.layers import premap_ids, remap_ids, remap_ids_v2
import numpy as np


def get_keys():
    return np.array([4, 2, 6, 1, 8, 7, 9, 3, 30])


def get_mapper():
    keys = get_keys()
    return {int(k): int(v) for v, k in enumerate(keys)}


def test_premap():
    mapper = get_mapper()
    keys = get_keys()
    premap = premap_ids(mapper, default=99)
    values, _ = premap(keys, False)
    assert all(values.squeeze() == np.asarray(range(len(keys))))


def test_remap():
    mapper = get_mapper()
    keys = get_keys()
    remap = remap_ids(mapper, default=99)
    values, _ = remap(keys, False)
    assert all(values.squeeze() == np.asarray(range(len(keys))))


def test_remap_v2():
    mapper = get_mapper()
    keys = get_keys()
    remap = remap_ids_v2(mapper, default=99)
    values, _ = remap(keys, False)
    assert all(values.squeeze() == np.asarray(range(len(keys))))


def test_remap_premap_eq():
    mapper = get_mapper()
    keys = get_keys()
    remap = remap_ids(mapper, default=99)
    remap_v2 = remap_ids_v2(mapper, default=99)
    premap = premap_ids(mapper, default=99)
    values1, _ = remap(keys, False)
    values2, _ = remap_v2(keys, False)
    values3, _ = premap(keys, False)
    assert (values1 == values2).all()
    assert (values2 == values3).all()


def test_column():
    mapper = get_mapper()
    keys = get_keys()
    idx = np.zeros((len(keys), 4), dtype="int")
    idx[:, 3] = keys
    remap_v2 = remap_ids_v2(mapper, column=3)
    premap = premap_ids(mapper, column=3)
    assert all(remap_v2(idx, False)[0].squeeze() == np.asarray(range(len(keys))))
    assert all(premap(idx, False)[0].squeeze() == np.asarray(range(len(keys))))
