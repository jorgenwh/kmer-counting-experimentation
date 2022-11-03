import random
import numpy as np

from accounters import get_reverse_complements
from accounters import convert_ACTG_to_ACGT_encoding

from accounters import get_ACGT_reverse_complement
from accounters import kmer_word_to_bitstring
from accounters import bitstring_to_ACGT
from accounters import word_reverse_complement

# Passes for all kmers
def assert_leading_A(kmers):
    mask = (1 << 63) | (1 << 62)
    size = kmers.shape[0]
    for i, kmer in enumerate(kmers):
        assert (kmer & mask) == 0
        print(f"{i}/{size}", end="\r")
    print(f"{size}/{size}")

def count_reverse_complements(kmers, num_checks=100):
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
    kmers = np.load("data/npy/uniquekmersACGT.npy")
    count = count_reverse_complements(kmers, num_checks=5000)
    print(count)

if __name__ == "__main__":
    main()
    
