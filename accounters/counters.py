import numpy as np
import cupy as cp

from accounters_C import CuHashTable, CppHashTable
from .kernels import _init_kernel, _count_kernel, _lookup_kernel


class CuCounter(CuHashTable):
    def __init__(self, keys, capacity: int = 0, capacity_factor: float = 1.75):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type"
        assert capacity_factor > 1.0, "capacity_factor must be greater than 1.0"

        # Dynamically determine hashtable capacity if not provided
        if capacity == 0:
            capacity = int(keys.size * capacity_factor) 

        assert capacity > keys.size, "Capacity must be greater than size of keyset"

        if isinstance(keys, np.ndarray):
            super().__init__(keys, capacity)
        elif isinstance(keys, cp.ndarray):
            super().__init__(keys.data.ptr, keys.size, capacity)

    def count(self, keys):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type"

        if isinstance(keys, np.ndarray):
            super().count(keys)
        elif isinstance(keys, cp.ndarray):
            super().count(keys.data.ptr, keys.size)

    def __getitem__(self, keys):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type"

        if isinstance(keys, np.ndarray):
            return super().get(keys)
        elif isinstance(keys, cp.ndarray):
            counts = cp.zeros_like(keys, dtype=np.uint32)
            super().get(keys.data.ptr, counts.data.ptr, keys.size)
            return counts 


class CppCounter(CppHashTable):
    def __init__(self, keys: np.ndarray, capacity: int = 0, capacity_factor: float = 1.75):
        assert isinstance(keys, np.ndarray), "Keys must be of type numpy.ndarray"
        assert capacity_factor > 1.0, "Capacity factor must be greater than 1.0"

        # Dynamically determine hashtable capacity if not provided
        if capacity == 0:
            capacity = int(keys.size * capacity_factor) 
        assert capacity > keys.size, "Capacity must be greater than size of keyset"

        super().__init__(keys, capacity)

    def count(self, keys, threads: int = 0):
        assert isinstance(keys, np.ndarray), "Keys must be of type numpy.ndarray"
        super().count(keys, threads)

    def __getitem__(self, keys, threads: int = 0):
        assert isinstance(keys, np.ndarray), "Keys must be of type numpy.ndarray"
        return super().get(keys)


class CupyCounter():
    def __init__(self, keys, capacity: int = 0, capacity_factor: int = 1.75):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type" 
        assert keys.dtype == np.uint64, "Keys must be of type uint64"
        keys = cp.asanyarray(keys, dtype=np.uint64)
        if len(keys.shape) > 1:
            keys = keys.reshape(-1)

        # Dynamically determine hashtable capacity if not provided
        if capacity == 0:
            capacity = int(keys.size * capacity_factor) 
        assert capacity > keys.size, "Capacity must be greater than size of keyset"

        self._capacity = capacity
        self._size = keys.size

        self._thread_block_size = 512

        self._keys = cp.full(capacity, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
        self._values = cp.full(capacity, 0xFFFFFFFF, dtype=np.uint32)

        _sz = keys.size
        grid_size = int(_sz / self._thread_block_size + (_sz % self._thread_block_size > 0))

        _init_kernel[grid_size, self._thread_block_size](
                self._keys, self._values, keys, _sz, self._capacity)

    def count(self, keys):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type" 
        assert keys.dtype == np.uint64, "Keys must be of type uint64"
        keys = cp.asanyarray(keys, dtype=np.uint64)
        if len(keys.shape) > 1:
            keys = keys.reshape(-1)
        
        _sz = keys.size
        grid_size = int(_sz / self._thread_block_size + (_sz % self._thread_block_size > 0))

        _count_kernel[grid_size, self._thread_block_size](
                self._keys, self._values, keys, _sz, self._capacity)

    def __getitem__(self, keys):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type" 
        assert keys.dtype == np.uint64, "Keys must be of type uint64"
        keys = cp.asanyarray(keys, dtype=np.uint64)
        if len(keys.shape) > 1:
            keys = keys.reshape(-1)
        counts = cp.zeros_like(keys, dtype=np.uint32)
        
        _sz = keys.size
        grid_size = int(_sz / self._thread_block_size + (_sz % self._thread_block_size > 0))

        _lookup_kernel[grid_size, self._thread_block_size](
                self._keys, self._values, keys, counts, _sz, self._capacity)

        return counts

    def __repr__(self):
        s = f"Counter({self._keys[:40]}, {self._values[:40]}, size={self._size}, capacity={self._capacity})"
        return s
                    
    def __str__(self):
        return self.__repr__()
