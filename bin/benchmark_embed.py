import nltk
import plac
import os
from os import path
import io
import gzip
from collections import defaultdict
import cProfile
import pstats

from thinc.neural.eeap import Embed
from thinc.neural.eeap import NumpyOps


def iter_files(giga_dir):
    i = 0
    for subdir in os.listdir(giga_dir):
        if not path.isdir(path.join(giga_dir, subdir)):
            continue
        for filename in os.listdir(path.join(giga_dir, subdir)):
            if filename.endswith('gz'):
                print(filename)
                yield path.join(giga_dir, subdir, filename)
                i += 1
                if i >= 1:
                    break
        break


def main(giga_dir):
    ops = NumpyOps()
    vectors = defaultdict(lambda: ops.allocate((300,)))
    W = ops.allocate((200, 300))
    embed = Embed(vectors=vectors, W=W, ops=ops)
    nr_word = 0
    for loc in iter_files(giga_dir):
        with gzip.open(loc, 'r') as file_:
            text = file_.read()
        words = text.split()
        vectors = embed.predict_batch(words)
        for word in words:
            if word not in embed.vectors:
                embed.vectors[word] = embed.ops.allocate((300,))
        nr_word += len(words)
    print(nr_word)


if __name__ == '__main__':
    if 0:
        plac.call(main)
    else:
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

        
