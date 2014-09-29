from .model cimport Model
from .thinc.features.extractor cimport Extractor


cdef class Perceptron:
    def __cinit__(self, templates, classes):
        self._extractor = Extractor(templates)
        self.classes = classes
        self.class_map = dict((clas, i) for i, clas in enumerate(sorted(classes)))
        self.nr_class = len(self.classes)
        self._model = LinearModel(self.nr_class)
        # Number of instances seen
        self.i = 0
        self.nr_correct = 0
        self.nr_total = 0

    @property
    def accuracy_string(self):
        acc = float(self.nr_correct) / self.nr_total
        return '%.2f' % (acc * 100)

    cdef fill_scores(self, double* scores, uint64_t* context):
        self._extractor.extract(self._features, context)
        self._model.score(scores, self._features)

    def end_training(self):
        pass

    def serialize(self, path):
        pass

    def deserialize(self, path):
        pass
