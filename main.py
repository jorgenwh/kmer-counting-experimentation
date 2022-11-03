import random
import numpy as np

from accounters import get_reverse_complements
from accounters import convert_ACTG_to_ACGT_encoding

from accounters import get_ACGT_reverse_complement
from accounters import kmer_word_to_bitstring
from accounters import bitstring_to_ACGT

from accounters import word_reverse_complement
from accounters import wrc
from accounters import test

def visual_sanity_check():
    kmers = np.load("data/npy/uniquekmersACGT.npy")
    while True:
        i = random.randint(0, kmers.shape[0])
        #kmer_size = random.randint(16, 32)
        kmer_size = 31

        kmer = int(kmers[i])
        kmer_bits = kmer_word_to_bitstring(kmer, kmer_size)
        kmer_seq = bitstring_to_ACGT(kmer_bits)
        expected_revcomp_seq = get_ACGT_reverse_complement(kmer_seq)

        revcomp = wrc(kmer, kmer_size)
        revcomp_bits = kmer_word_to_bitstring(revcomp, kmer_size)
        revcomp_seq = bitstring_to_ACGT(revcomp_bits)

        print("\n"*50)
        print(f"kmer_size={kmer_size}")
        print("\nkmer and revcomp bits")
        print(kmer_bits)
        print(revcomp_bits)
        print("\nkmer and revcomp ACGT sequences")
        print(kmer_seq)
        print(revcomp_seq)
        print(expected_revcomp_seq)
        inp = input("\naction: ")
        if inp == "q":
            break

# Passes for all kmers
def assert_leading_A(kmers):
    kmers = np.load("data/npy/uniquekmersACGT.npy")

    mask = (1 << 63) | (1 << 62)
    size = kmers.shape[0]
    for i, kmer in enumerate(kmers):
        assert (kmer & mask) == 0
        print(f"{i}/{size}", end="\r")
    print(f"{size}/{size}")

def count_reverse_complements(kmers, num_checks=100):
    kmers = np.load("data/npy/uniquekmersACGT.npy")
    count = 0
    size = kmers.shape[0]

    revcomps = get_reverse_complements(kmers, 31)

    for _ in range(num_checks):
        i = random.randint(0, size)

        kmer = int(kmers[i])
        if revcomp in kmers:
            count += 1

        print(f"{_}/{num_checks}, count={count}", end="\r")
    print(f"{num_checks}/{num_checks}, count={count}")

    return count
        
def main():
    visual_sanity_check()

if __name__ == "__main__":
    main()
    
