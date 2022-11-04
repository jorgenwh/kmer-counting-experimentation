import numpy as np

from accounters.kmers import word_reverse_complement
from accounters import word_reverse_complement_C 
from accounters import get_reverse_complements

from accounters import kmer_word_to_bitstring
from accounters import bitstring_to_ACGT
from accounters import get_ACGT_reverse_complement

def test_word_reverse_complement():
    word_bits = "0101100110010101101000100001111001001001010111001001011100000001"
    word = int(word_bits, 2)
    revcomp_bits = "1011111100101001110010101001111001001011011101011010100110011010"
    
    # Test different kmer sizes
    for kmer_size in range(16, 32+1):
        revcomp_py = word_reverse_complement(word, kmer_size)
        revcomp_C = word_reverse_complement_C(word, kmer_size)
        revcomp_expected = int("00"*kmer_size + revcomp_bits[:kmer_size*2], 2)
        assert revcomp_py == revcomp_C == revcomp_expected

    for kmer_size in range(16, 32+1):
        num_kmers = 10
        kmers = np.random.randint(low=0, high=0xFFFFFFFFFFFFFFFF, size=num_kmers, dtype=np.uint64)
        revcomps = get_reverse_complements(kmers, kmer_size)

        for i in range(num_kmers):
            kmer = int(kmers[i])
            revcomp = int(revcomps[i])
            revcomp_py = word_reverse_complement(kmer, kmer_size)
            revcomp_C = word_reverse_complement_C(kmer, kmer_size) 
            assert revcomp == revcomp_py == revcomp_C

def test_bitstring_to_ACGT():
    bitstring1 = "0101100110010101101000100001111001001001010111001001011100000001"
    bitstring2 = "[0101]100110010101101000100001111001001001010111001001011100000001"
    ACGT_seq1 = bitstring_to_ACGT(bitstring1)
    ACGT_seq2 = bitstring_to_ACGT(bitstring2)
    expected_ACGT_seq1 = "CCGCGCCCGGAGACTGCAGCCCTAGCCTAAAC"
    expected_ACGT_seq2 = "[CC]GCGCCCGGAGACTGCAGCCCTAGCCTAAAC"
    assert ACGT_seq1 == expected_ACGT_seq1
    assert ACGT_seq2 == expected_ACGT_seq2

def test_ACGT_reverse_complement():
    ACGT_seq1 = "CCGCGCCCGGAGACTGCAGCCCTAGCCTAAAC"
    ACGT_seq2 = "[CCGC]GCCCGGAGACTGCAGCCCTAGCCTAAAC"
    revcomp1 = get_ACGT_reverse_complement(ACGT_seq1)
    revcomp2 = get_ACGT_reverse_complement(ACGT_seq2)
    expected_revcomp1 = "GTTTAGGCTAGGGCTGCAGTCTCCGGGCGCGG"
    expected_revcomp2 = "[AAAA]GTTTAGGCTAGGGCTGCAGTCTCCGGGC"
    assert revcomp1 == expected_revcomp1
    assert revcomp2 == expected_revcomp2

def test_revcomp():
    word_bits = "0101100110010101101000100001111001001001010111001001011100000001"
    word = int(word_bits, 2)
    revcomp_bits = "1011111100101001110010101001111001001011011101011010100110011010"
    
    revcomp = word_reverse_complement(word, 32) 
    assert kmer_word_to_bitstring(revcomp, 32) == revcomp_bits

    word_ACGT_seq = bitstring_to_ACGT(word_bits)
    revcomp_ACGT_seq = bitstring_to_ACGT(revcomp_bits)
    assert get_ACGT_reverse_complement(word_ACGT_seq) == revcomp_ACGT_seq
