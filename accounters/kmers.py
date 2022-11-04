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
    for i in range(63, -1, -1):
        bit = (1 << i) & int(kmer)
        bits += str(int(bool(bit)))

    if empty_bits:
        bits = "[" + bits[:empty_bits] + "]" + bits[empty_bits:]

    return bits

def bitstring_to_ACGT(bitstring):
    if "[" not in bitstring and "]" not in bitstring:
        return "".join([_bit_bases[bitstring[i:(i+2)]] for i in range(0, len(bitstring), 2)])

    discarded_bases = "".join([_bit_bases[bitstring[i:(i+2)]] for i in range(bitstring.index("[")+1, bitstring.index("]"), 2)])
    bases = "".join([_bit_bases[bitstring[i:(i+2)]] for i in range(bitstring.index("]")+1, len(bitstring), 2)])
    return "[" + discarded_bases + "]" + bases

def get_ACGT_reverse_complement(ACGT_sequence):
    if "[" not in ACGT_sequence and "]" not in ACGT_sequence:
        return "".join([_complementary_bases[ACGT_sequence[i]] for i in range(len(ACGT_sequence)-1, -1, -1)])

    discarded_bases = "A" * ((ACGT_sequence.index("]") - ACGT_sequence.index("[")) - 1)
    bases = "".join([_complementary_bases[ACGT_sequence[i]] for i in range(len(ACGT_sequence)-1, ACGT_sequence.index("]"), -1)])
    return "[" + discarded_bases + "]" + bases

