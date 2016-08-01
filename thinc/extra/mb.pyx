from libc.string cimport memcpy

from .eg cimport Example

cdef class Minibatch:
    def __dealloc__(self):
        del self.c

    def __len__(self):
        return self.c.i

    def __getitem__(self, int i):
        cdef Example eg = Example(nr_class=self.nr_class, nr_feat=self.c.nr_feat(i))
        memcpy(eg.c.features,
            self.c.features(i), eg.nr_feat * sizeof(eg.c.features[0]))
        memcpy(eg.c.scores,
            self.c.scores(i), eg.c.nr_class * sizeof(eg.c.scores[0]))
        memcpy(eg.c.costs,
            self.c.costs(i), eg.c.nr_class * sizeof(eg.c.costs[0]))
        memcpy(eg.c.is_valid,
            self.c.is_valid(i), eg.c.nr_class * sizeof(eg.c.is_valid[0]))
        return eg

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def widths(self):
        return [self.c.widths[i] for i in range(self.c.nr_layer)]

    @property
    def nr_class(self):
        return self.c.nr_out()

    def guess(self, i):
        return self.c.guess(i)

    def best(self, i):
        return self.c.best(i)

    def loss(self, i):
        return 1.0 - self.c.scores(i)[self.c.best(i)]

