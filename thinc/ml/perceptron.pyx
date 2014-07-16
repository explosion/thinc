from .model cimport Model
from .thinc.features.extractor cimport Extractor


cdef class Perceptron:
    def __cinit__(self, templates, classes):
        self._extractor = Extractor(templates)
        self._model = Model(nr_class)
        self.classes = classes
        self.class_map = dict((clas, i) for i, clas in enumerate(sorted(classes)))
        self.nr_class = len(self.classes)
        # Number of instances seen
        self.i = 0
        self.nr_correct = 0
        self.nr_total = 0

    @property
    def accuracy_string(self):
        acc = float(self.nr_correct) / self.nr_total
        return '%.2f' % (acc * 100)

    def score(self, context):
        self._extractor.extract(self._features, context)
        self._model.score(self._scores, features)
        scores = {}
        for clas in self.classes:
            scores[clas] = self._scores[self.class_map[clas]]
        return scores

    def best_from(self, classes):
        pass

    def tell_answer(self, truths):
        truth = self.best_from(truths)

    def end_training(self):
        pass

    def serialize(self, path):
        pass

    def deserialize(self, path):
        pass
