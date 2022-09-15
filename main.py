import sys
import time
import math
import argparse

import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

def time_hash(verbose=False):
    n = 0
    t1 = time.time()

    for chunk in bnp.open(sys.argv[1]):
        name = chunk.name
        sequence = chunk.sequence # raggedarray where each row is a sequence of len 150 (a read)

        kmers = bnp.kmers.fast_hash(sequence, 31, bnp.encodings.ACTGEncoding)

        n += 1
        if n >= 100:
            break

        if verbose:
            print(f"chunks processed: {n}", end="\r")

    t2 = time.time()
    if verbose:
        print(f"chunks processed: {n}")
    return t2 - t1

if __name__ == "__main__":
    #time_hash(verbose=False)

    unique_kmers = np.load("uniquekmers.npy")[:50000000]

    counter = nps.Counter(keys=unique_kmers)
    lengths = counter._keys.shape.lengths
    print(np.amax(lengths))
