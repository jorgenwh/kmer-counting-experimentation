import random
import numpy as np

from accounters import CuCounter

from accounters import get_reverse_complements
from accounters import convert_ACTG_to_ACGT_encoding

from accounters import get_ACGT_reverse_complement
from accounters import kmer_word_to_bitstring
from accounters import bitstring_to_ACGT

from accounters import word_reverse_complement
from accounters import word_reverse_complement_C
from accounters import test

def visual_sanity_check(kmers):
    while True:
        i = random.randint(0, kmers.shape[0])
        kmer_size = random.randint(16, 32)
        #kmer_size = 31

        kmer = int(kmers[i])
        kmer_bits = kmer_word_to_bitstring(kmer, kmer_size)
        kmer_seq = bitstring_to_ACGT(kmer_bits)
        expected_revcomp_seq = get_ACGT_reverse_complement(kmer_seq)

        revcomp = word_reverse_complement(kmer, kmer_size)
        revcomp_bits = kmer_word_to_bitstring(revcomp, kmer_size)
        revcomp_seq = bitstring_to_ACGT(revcomp_bits)

        print("\n"*75)
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
    mask = (1 << 63) | (1 << 62)
    size = kmers.shape[0]
    for i, kmer in enumerate(kmers):
        assert (kmer & mask) == 0
        print(f"{i}/{size}", end="\r")
    print(f"{size}/{size}")

def assert_no_revcomps(kmers, kmer_size):
    for i, kmer in enumerate(kmers):
        revcomp = word_reverse_complement(int(kmer), kmer_size)
        assert revcomp not in kmers
        print(f"{i}/{kmers.shape[0]}", end="\r")
    print(f"{kmers.shape[0]}/{kmers.shape[0]}")

def count_reverse_complements(kmers, num_checks=100):
    count = 0
    size = kmers.shape[0]

    for _ in range(num_checks):
        i = random.randint(0, size)

        kmer = int(kmers[i])
        revcomp = np.uint64(word_reverse_complement(kmer, 31))

        if revcomp in kmers:
            count += 1

        print(f"{_}/{num_checks}, count={count}", end="\r")
    print(f"{num_checks}/{num_checks}, count={count}")

    return count

def create_random_dataset(num_kmers=1000000, revcomp_prob=0.8):
    kmers = []
    while len(kmers) < num_kmers:
        kmer = int(np.random.randint(0, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64))
        kmer &= 0x3FFFFFFFFFFFFFFF
        revcomp = word_reverse_complement(kmer, 31)
        if kmer not in kmers and revcomp not in kmers:
            kmers.append(kmer)

    kmers = np.array(kmers, dtype=np.uint64)

    revcomps = np.random.choice(kmers, size=int(num_kmers*revcomp_prob), replace=False)
    revcomps = get_reverse_complements(revcomps, 31)
    assert revcomps.dtype == kmers.dtype == np.uint64

    all_kmers = np.concatenate((kmers, revcomps))
    unique_kmers = np.unique(all_kmers)
    assert unique_kmers.dtype == np.uint64

    #np.save("data/npy/uniquekmers_randgen.npy", unique_kmers)
    return kmers, revcomps, unique_kmers
        
def main():
    keys, _, unique_kmers = create_random_dataset(num_kmers=1000, revcomp_prob=0.8)
    #keys = np.array([int("0001111010011010111011011100010101110100110000000011010100011011", 2)], dtype=np.uint64)
    revcomps = get_reverse_complements(keys, 31)

    #print(f"{keys[0]} -> {revcomps[0]}\n")

    print("Asserting keys contain no revcomps...")
    assert_no_revcomps(keys, 31)
    print("Passed\n")

    counter = CuCounter(keys=keys)
    #print(counter, end="\n\n")

    # Counts share indices with keys
    counts = np.zeros_like(keys)

    chunk_size = 1000
    kmers = np.random.choice(unique_kmers, size=chunk_size)
    #kmers = revcomps 

    #print(f"Counting kmers = {kmers}, with revcomps = {get_reverse_complements(kmers, 31)}")
    counter.count(kmers)  
    #print(counter)

    for kmer in kmers:
        revcomp = word_reverse_complement(int(kmer), 31)
        for i in range(len(keys)):
            if keys[i] == kmer or keys[i] == revcomp:
                counts[i] += 1
                break

    print("\nAsserting counts...")
    np.testing.assert_equal(counter[keys], counts) 
    print("Passed")


if __name__ == "__main__":
    main()
    
