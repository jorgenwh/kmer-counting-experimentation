import numpy as np
import cupy as cp

from accounters import kmer_hashes_to_ascii 

kmers = np.load("data/npy/uniquekmers.npy")[0]
print(f"kmers.shape={kmers.shape}, kmers.dtype={kmers.dtype}")

converted = kmer_hashes_to_ascii(kmers)
size = converted.size
print(f"converted.shape={converted.shape}, converted.dtype={converted.dtype}")
print(converted)
print()
print(converted)
print()
print()

x1 = np.array([chr(x) for x in range(127)])[converted]
x2 = np.array([chr(x) for x in range(127)])[converted]

print(x1)
print()
print(x2)
