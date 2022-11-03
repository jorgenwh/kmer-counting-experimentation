_base_bits = {"A": "00", "C": "01", "G": "10", "T": "11"}
_bit_bases = {"00": "A", "01": "C", "10": "G", "11": "T", "--": "-"}
_complementary_bases = {"A": "T", "C": "G", "G": "C", "T": "A"}

def word_reverse_complement(kmer, kmer_size):
    res = ~kmer;
    res = ((res >> 2 & 0x3333333333333333) | (res & 0x3333333333333333) << 2);
    res = ((res >> 4 & 0x0F0F0F0F0F0F0F0F) | (res & 0x0F0F0F0F0F0F0F0F) << 4);
    res = ((res >> 8 & 0x00FF00FF00FF00FF) | (res & 0x00FF00FF00FF00FF) << 8);
    res = ((res >> 16 & 0x0000FFFF0000FFFF) | (res & 0x0000FFFF0000FFFF) << 16);
    res = ((res >> 32 & 0x00000000FFFFFFFF) | (res & 0x00000000FFFFFFFF) << 32);
    return (res >> (2 * (32 - kmer_size)));

def kmer_word_to_bitstring(kmer, kmer_size):
    empty_bits = 2 * (32 - kmer_size)
    bits = ""
    for i in range(63 - empty_bits, -1, -1):
        bit = (1 << i) & int(kmer)
        bits += str(int(bool(bit)))
    bits = ("-" * empty_bits) + bits
    return bits

def bitstring_to_ACGT(bitstring):
    ACGT_sequence = ""
    for i in range(0, len(bitstring), 2):
        bits = bitstring[i:(i+2)]
        base = _bit_bases[bits]
        ACGT_sequence += base
    return ACGT_sequence

def get_ACGT_reverse_complement(ACGT_sequence):
    revcomp = ""
    b = 0
    while ACGT_sequence[b] == "-":
        revcomp += "-"
        b += 1

    empty_bases = b
    for i in range(empty_bases, len(ACGT_sequence)):
        base = ACGT_sequence[i]
        revcomp += _complementary_bases[base]
    
    revcomp = ("-" * empty_bases) + revcomp[::-1]
    return revcomp

