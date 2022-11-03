import random

import numpy as np
import cupy as cp

from accounters import get_reverse_complements
from accounters import get_unique_complements

from accounters import get_ACGT_reverse_complement
from accounters import kmer_word_to_bitstring
from accounters import bitstring_to_ACGT

def test_start_is_empty():
    kmers = np.load("data/npy/uniquekmersACGT.npy")[:5000000]
    for i, kmer in enumerate(kmers):
        bitstring = kmer_word_to_bitstring(kmer, 32)
        seq = bitstring_to_ACGT(bitstring)
        assert seq[:2] == "AA"

        print(f"{i}/{kmers.shape[0]}", end="\r")
    print(f"{i+1}/{kmers.shape[0]}")

def main():
    KMER_SIZE = 31
    kmers = np.load("data/npy/uniquekmersACGT.npy")[1000000:1000001]

    revcomps = get_reverse_complements(kmers, KMER_SIZE)
    assert kmers.shape == revcomps.shape and kmers.dtype == revcomps.dtype

    kmer = kmers[0]
    revcomp = revcomps[0]

    full_kmer_bitstring = kmer_word_to_bitstring(kmer, 32)
    print(len(full_kmer_bitstring))
    kmer_bitstring = kmer_word_to_bitstring(kmer, KMER_SIZE)
    revcomp_bitstring = kmer_word_to_bitstring(revcomp, KMER_SIZE)

    print(f"full_kmer_bitstring:\n{full_kmer_bitstring}")
    print(f"kmer_bitstring:\n{kmer_bitstring}")
    print(f"revcomp_bitstring:\n{revcomp_bitstring}\n")

    kmer_ACGT_seq = bitstring_to_ACGT(kmer_bitstring)
    revcomp_ACGT_seq = bitstring_to_ACGT(revcomp_bitstring)
    expected_revcomp_ACGT_seq = get_ACGT_reverse_complement(kmer_ACGT_seq)

    print(f"kmer_seq:\n{kmer_ACGT_seq}")
    print(f"revcomp_seq:\n{revcomp_ACGT_seq}")
    print(f"expected_revcomp_seq:\n{expected_revcomp_ACGT_seq}\n")

    assert revcomp_ACGT_seq == expected_revcomp_ACGT_seq
    print("All asserts passed")

if __name__ == "__main__":
    main()
