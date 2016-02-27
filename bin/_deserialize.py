'''Load a spaCy model

1. Download and extract https://index.spacy.io/models/en_default-1.0.8/archive.gz
2. Find the model file deps/model
3. Pass the path to this file to this script
'''

from __future__ import absolute_import

import sys
import thinc.linear.avgtron


def main(loc):
    model = thinc.linear.avgtron.AveragedPerceptron([])
    model.load(loc)


if __name__ == '__main__':
    main(sys.argv[1])
