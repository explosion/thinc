from numpy import prod


class Params(object):
    def __init__(self, ops, size=128):
        if size < 0:
            raise ValueError("TODO error re negative size %d" % size)
        self.ops = ops
        self._mem = self.ops.allocate((1, size))
        self._offsets = {}
        self._i = 0

    def get(self, name):
        if name.startswith('d_'):
            name = name[2:]
            if self._mem.shape[0] == 1:
                self._alloc_gradients()
            col = 1
        else:
            col = 0
        if name not in self._offsets:
            return None
        offset, shape = self._offsets[name]
        return self._mem[col, offset : offset + prod(shape)].reshape(shape)

    def add(self, name, shape):
        self._offsets[name] = (self._i, shape)
        blob = self._get_blob(prod(shape))
        return blob.reshape(shape)

    def _get_blob(self, nr_req):
        nr_avail = self._mem.shape[1] - self._i
        if nr_avail < nr_req:
            self._realloc((self._mem.shape[1] + nr_req) * 2)
        blob = self._mem[:, self._i : self._i + nr_req]
        self._i += nr_req
        return blob

    def _alloc_gradients(self):
        new_mem = self.ops.allocate((2, self._mem.shape[1]))
        new_mem[0] = self._mem[0]
        self._mem = new_mem

    def _realloc(self, new_size):
        new_mem = self.ops.allocate((self._mem.shape[0], new_size))
        new_mem[:, :self._i+1] = self._mem[:, :self._i+1]
        self._mem = new_mem
