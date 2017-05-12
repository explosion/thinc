from libc.string cimport memcpy
from libc.stdlib cimport calloc, free
from ..typedefs cimport len_t

from .eg cimport Example

cdef class Minibatch:
    def __cinit__(self, nr_class=None, widths=None, batch_size=0):
        if widths is None and nr_class is None:
            nr_class = 1
        if widths is None:
            widths = [nr_class]
        self.c = NULL
        if widths != None:
            c_widths = <len_t*>calloc(len(widths), sizeof(len_t))
            for i, width in enumerate(widths):
                c_widths[i] = width
            self.c = new MinibatchC(c_widths, len(widths), batch_size)
            free(c_widths)
        else:
            self.c = new MinibatchC(NULL, 0, batch_size)

    def __dealloc__(self):
        if self.c != NULL:
            del self.c
        self.c = NULL

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

