import random

import numpy as np
import cupy as cp

from accounters import kmer_hashes_to_ascii, ascii_to_kmer_hashes, get_reverse_complements

base_bits = {"A": "00", "C": "01", "G": "10", "T": "11"}
bit_bases = {"00": "A", "01": "C", "10": "G", "11": "T", "--": "-"}
comp_bases = {"A": "T", "C": "G", "T": "A", "G": "C"}


def to_bitstring(kmer, size):
    bits = "-" * 2 * (32 - size)
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

def get_revcomp(ACTG_seq):
    revcomp = ""
    for b in ACTG_seq[(32-KMER_SIZE)*2:]:
        revcomp += comp_bases[b]

KMER_SIZE = 31
kmers = np.load("data/npy/uniquekmers.npy")[1000000:1000002]
print(kmers)

revcomps = get_reverse_complements(kmers, KMER_SIZE)
print(revcomps)

kmer_bitstring = to_bitstring(kmers[0])
kmer_seq = bitstring_to_ACTG(kmer_bitstring)

expected_revcomp_seq = get_revcomp(kmer_seq)

revcomp_bitstring = to_bitstring(revcomps[0])
revcomp_seq = bitstring_to_ACTG(revcomp_bitstring)

print("\nbitstrings")
print(f"kmer:\n{kmer_bitstring}")
print(f"revcomp:\n{revcomp_bitstring}\n")

print("ACGT sequences")
print(f"kmer:\n{kmer_seq}")
print(f"revcomp:\n{revcomp_seq}")
print(f"expected:\n{expected_revcomp_seq}")
