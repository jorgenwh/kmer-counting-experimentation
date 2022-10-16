import numpy as np
import cupy as cp

from accounters import get_unique_complements, kmers_to_strings

kmers = np.load("data/npy/uniquekmers.npy")
print(kmers.shape)
kmer_strings = kmer_strings(kmers)
print(kmer_strings.shape)

#print(keys.shape)
#unique_complements = get_unique_complements(keys)
#print(unique_complements.shape)
