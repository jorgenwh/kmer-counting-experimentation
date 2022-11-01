import random

import numpy as np
import cupy as cp

from accounters import kmer_hashes_to_ascii, ascii_to_kmer_hashes, get_reverse_complements

base_bits = {"A": "00", "C": "01", "T": "10", "G": "11"}
bit_bases = {"00": "A", "01": "C", "10": "T", "11": "G"}

def to_bitstring(kmer):
    bits = ""
    for i in range(64):
        bit = int(bool((1 << i) & int(kmer)))
        bits += str(bit)
    bits = bits[::-1]
    return bits

def bitstring_to_ACTG(bitstring):
    seq = ""
    for i in range(0, len(bitstring), 2):
        bits = bitstring[i:i+2]
        base = bit_bases[bits]
        seq += base
    return seq


kmers = np.load("data/npy/uniquekmers.npy")[1000000:1000002]
print(kmers)
revcomps = get_reverse_complements(kmers, 31)

kmer_bitstring = to_bitstring(kmers[0])
kmer_seq = bitstring_to_ACTG(kmer_bitstring)

revcomp_bitstring = to_bitstring(revcomps[0])
revcomp_seq = bitstring_to_ACTG(revcomp_bitstring)

print("\nbitstrings")
print(f"kmer:\n{kmer_bitstring}")
print(f"revcomp:\n{revcomp_bitstring}\n")

print("ACGT sequences")
print(f"kmer:\n{kmer_seq}")
print(f"revcomp:\n{revcomp_seq}")
