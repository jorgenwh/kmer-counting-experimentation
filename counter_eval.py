import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

if __name__ == "__main__":
    # total of 1.2GB
    unique_kmers = np.load("data/npy/uniquekmers.npy")[:50000000]
    count_kmers = np.load("data/npy/testkmers.npy")[:100000000]

    counter = nps.Counter(keys=unique_kmers, value_dtype=np.uint64)
    counter.count(count_kmers)

    assert counter._keys._data.shape == counter._values._data.shape, f"{counter._keys._data.shape} != {counter._values._data.shape}"
    assert counter._keys._data.dtype == counter._values._data.dtype, f"{counter._keys._data.dtype} != {counter._values._data.dtype}"
    assert counter._keys._data.dtype == np.uint64, "keys.dtype != np.uint64"

    #for k, v in zip(counter._keys.ravel(), counter._values.ravel()):
        #pass

    counter._keys._data.tofile("keys.bin")
    counter._values._data.tofile("values.bin")
