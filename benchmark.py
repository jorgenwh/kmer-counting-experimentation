import time

import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

def pipeline(filename):
    t1 = time.time()

    chunk_generator = bnp.open(filename)
    print(type(chunk_generator))

    t2 = time.time()
    print(f"Chunk generator initialization: {round(t2 - t1, 1)} seconds")

    t1 = time.time()

    n = 0
    for chunk in chunk_generator:
        n += 1

    t2 = time.time()
    print(f"Iterating over chunks: {round(t2 - t1, 1)} seconds")
    print(f"Mean chunk time: {round((t2 - t1)/n, 4)} seconds")

if __name__ == "__main__":
    filename = "data/fa/testreads20m.fa"

    xp = np

    if xp.__name__ == "cupy":
        nps.set_backend(xp)
        bnp.set_backend(xp)

    print(f"Array module: {xp.__name__}")

    pipeline(filename)

