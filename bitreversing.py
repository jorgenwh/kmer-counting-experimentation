import numpy as np

def get_bits(integer, size):
    bits = ""
    for i in range(size-1, -1, -1):
        bit = (1 << i) & int(integer)
        bits += str(int(bool(bit)))
    return bits

def print_bits(integer, size):
    bits = ""
    for i in range(size-1, -1, -1):
        bit = (1 << i) & int(integer)
        bits += str(int(bool(bit)))
    print(bits)

"""
def _reverse_bits(b):
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1
    return b
"""

def _reverse_bits(b):
    b = (b & 0b0000) >> 4 | (b & 0b1111) << 4
    b = (b & 0b1100) >> 2 | (b & 0b0011) << 2
    b = (b & 0b1010) >> 1 | (b & 0b0101) << 1
    return b

def reverse_bits(arr):
    assert arr.dtype == np.uint64

    #_lookup = np.array([
        #0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 
        #0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf], dtype=np.uint8)
    _lookup = np.array([
        0b0000, 0b1000, 0b0100, 0b1100, 0b0010, 0b1010, 0b0110, 0b1110,
        0b0001, 0b1001, 0b0101, 0b1101, 0b0011, 0b1011, 0b0111, 0b1111], dtype=np.uint8)

    arr = arr.view(np.uint8)

    reversed_bytes = (_lookup[arr & 0b1111] << 4) | _lookup[arr >> 4]
    reversed_bits = np.copy(np.flip(reversed_bytes))
    return reversed_bits.view(np.uint64)

#x = np.random.randint(low=0, high=0xFF, size=1, dtype=np.uint8)

x = np.random.randint(low=0, high=0xFFFFFFFFFFFFFFFF, size=1, dtype=np.uint64)
original_bits = get_bits(x[0], 64)

y = reverse_bits(x)

reversed_bits = get_bits(y[0], 64)
assert reversed_bits[::-1] == original_bits
print("Assert passed")

