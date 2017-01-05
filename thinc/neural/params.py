from numpy import prod


class Params(object):
    def __init__(self, ops, size=128):
        if size < 0:
            raise ValueError("TODO error re negative size %d" % size)
        self.ops = ops
        self._mem = self.ops.allocate((2, size))
        self._offsets = {}
        self._i = 0
        self.allow_resize = True

    @property
    def weights(self):
        return self._mem[0, :self._i]

    @property
    def gradient(self):
        return self._mem[1, :self._i]

    @property
    def W(self):
        W = self.get('W', require=True)
        return W

    @property
    def b(self):
        b = self.get('b', require=True)
        return self.get('b')

    @property
    def d_W(self):
        d_W = self.get('d_W', require=True)
        return d_W
    
    @property
    def d_b(self):
        d_b = self.get('d_b', require=True)
        return d_b

    def __contains__(self, name):
        return name in self._offsets

    def get(self, name, require=False):
        if name.startswith('d_'):
            name = name[2:]
            col = 1
        else:
            col = 0
        if name not in self._offsets:
            if require:
                raise KeyError("TODO error")
            else:
                return None
        offset, shape = self._offsets[name]
        return self._mem[col, offset : offset + prod(shape)].reshape(shape)

    def add(self, name, shape):
        assert name not in self._offsets, "TODO error"
        assert not name.startswith('d_')
        self._offsets[name] = (self._i, shape)
        blob = self._get_blob(prod(shape))
        return blob[0].reshape(shape)

    def merge_params(self, others):
        others = list(others)
        if not others:
            return None
        if not all(other.allow_resize for other in others):
            raise ValueError("TODO Error")
        sizes = [other._i+1 for other in others]
        nr_req = self._i + sum(sizes)
        if self._mem.shape[1] < nr_req:
            self._realloc(nr_req)
        self.allow_resize = False
        for other in others:
            other.replace_mem(self._get_blob(other._i))
    
    def replace_mem(self, mem):
        if not self.allow_resize:
            raise ValueError("TODO Error")
        self.allow_resize = False
        mem[:] = self._mem[:, :self._i]
        self._mem = mem

    def _get_blob(self, nr_req):
        print("Alloc", nr_req)
        nr_avail = self._mem.shape[1] - (self._i+1)
        if nr_avail < nr_req:
            self._realloc(max(self._mem.shape[1], nr_req) * 2)
        blob = self._mem[:, self._i : self._i + nr_req]
        self._i += nr_req
        return blob

    def _realloc(self, new_size):
        if not self.allow_resize:
            raise ValueError("TODO Error")
        print("Realloc", self._i, new_size)
        new_mem = self.ops.allocate((self._mem.shape[0], new_size))
        new_mem[:, :self._i+1] = self._mem[:, :self._i+1]
        self._mem = new_mem
