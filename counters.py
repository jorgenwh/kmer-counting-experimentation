import numpy as np
import cupy as cp

from temp.cuht_module import NaiveHashTable, COPSHashTable

class NaiveCounter(NaiveHashTable):
    def __init__(self, keys, capacity=200000000):
        assert isinstance(keys, (np.ndarray, cp.ndarray))
        if isinstance(keys, np.ndarray):
            super().__init__(keys, capacity)
        elif isinstance(keys, cp.ndarray):
            super().__init__(keys.data.ptr, keys.size, capacity)

    def count(self, kmers):
        if isinstance(kmers, np.ndarray):
            super().count(kmers)
        elif isinstance(kmers, cp.ndarray):
            super().countcu(kmers.data.ptr, kmers.size)
        else:
            print("Error: invalid type provided as kmers")

    def __getitem__(self, keys):
        if isinstance(keys, np.ndarray):
            return super().get(keys)
        elif isinstance(keys, cp.ndarray):
            counts = cp.zeros_like(keys, dtype=np.uint32)
            super().getcu(keys.data.ptr, counts.data.ptr, keys.size)
            return counts 
        else:
            print("Error: invalid type provided as keys")


class COPSCounter(COPSHashTable):
    def __init__(self, keys, capacity=200000000, cg_size=4):
        assert isinstance(keys, (np.ndarray, cp.ndarray))
        if isinstance(keys, np.ndarray):
            super().__init__(keys, capacity, cg_size)
        elif isinstance(keys, cp.ndarray):
            super().__init__(keys.data.ptr, capacity, cg_size)

    def count(self, kmers):
        if isinstance(kmers, np.ndarray):
            super().count(kmers)
        elif isinstance(kmers, cp.ndarray):
            super().countcu(kmers.data.ptr, kmers.size)
        else:
            print("Error: invalid type provided as kmers")

    def __getitem__(self, keys):
        if isinstance(keys, np.ndarray):
            return super().get(keys)
        elif isinstance(keys, cp.ndarray):
            counts = cp.zeros_like(keys, dtype=np.uint32)
            super().getcu(keys.data.ptr, counts.data.ptr, keys.size)
            return counts 
        else:
            print("Error: invalid type provided as keys")
