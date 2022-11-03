import random

import numpy as np
import cupy as cp

from accounters import get_reverse_complements
from accounters import get_unique_complements

from accounters import get_ACGT_reverse_complement
from accounters import kmer_word_to_bitstring
from accounters import bitstring_to_ACGT

kmers = np.load("data/npy/uniquekmersACGT.npy")
for i, kmer in enumerate(kmers):
    bitstring = kmer_word_to_bitstring(kmer, 32)
    seq = bitstring_to_ACGT(bitstring)
    assert seq[-2:] == "AA"

    print(f"{i}/{kmers.shape[0]}", end="\r")
print(f"{i+1}/{kmers.shape[0]}")

"""
KMER_SIZE = 32
kmers = np.load("data/npy/uniquekmersACGT.npy")[1000000:1000001]
#kmers = np.load("data/npy/uniquekmersACGT.npy")

revcomps = get_reverse_complements(kmers, KMER_SIZE)
assert kmers.shape == revcomps.shape and kmers.dtype == revcomps.dtype

kmer = kmers[0]
revcomp = revcomps[0]

kmer_bitstring = kmer_word_to_bitstring(kmer, KMER_SIZE)
revcomp_bitstring = kmer_word_to_bitstring(revcomp, KMER_SIZE)

print(kmer_bitstring)
print(f"len={len(kmer_bitstring)}")
print(revcomp_bitstring)
print(f"len={len(revcomp_bitstring)}")

kmer_ACGT_seq = bitstring_to_ACGT(kmer_bitstring)
revcomp_ACGT_seq = bitstring_to_ACGT(revcomp_bitstring)

print(kmer_ACGT_seq)
print(revcomp_ACGT_seq)
"""
