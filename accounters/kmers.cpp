#include <stdexcept>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <bitset>

#include "kmers.h"

uint8_t shifts8b[4] = {
  6, 4, 2, 0
};

uint8_t shifts64b[32] = {
  62, 60, 58, 56, 54, 
  52, 50, 48, 46, 44, 
  42, 40, 38, 36, 34, 
  32, 30, 28, 26, 24, 
  22, 20, 18, 16, 14, 
  12, 10, 8, 6, 4, 2, 0
};

void hashes_to_ascii(const uint64_t *hashes, const uint64_t num_hashes, 
    uint8_t *bases, const uint64_t num_bases) {
  for (uint64_t i = 0; i < num_bases; i++) {
    uint8_t shift = shifts64b[i%32];
    uint64_t mask = BASE_MASK << shift;

    uint64_t bitbase = (hashes[i/32] & mask) >> shift;
    switch (bitbase) {
      case BASE_A:
        bases[i] = ASCII_A;
        break;
      case BASE_C:
        bases[i] = ASCII_C;
        break;
      case BASE_G:
        bases[i] = ASCII_G;
        break;
      case BASE_T:
        bases[i] = ASCII_T;
        break;
      default:
        throw std::runtime_error("invalid DNA base");
    }
  }
}

void encode_kmers(const char *bases, const uint32_t num_kmers, uint64_t *kmers, const uint32_t kmer_size) {
  uint64_t num_bases = num_kmers * kmer_size;

  for (uint64_t i = 0; i < num_bases; i++) {
    uint8_t shift = shifts64b[i%32];
    uint64_t bitbase;

    switch (bases[i]) {
      case 'A': case 'a':
        bitbase = uint64_t(BASE_A) << shift;
        break;
      case 'C': case 'c':
        bitbase = uint64_t(BASE_C) << shift;
        break;
      case 'G': case 'g':
        bitbase = uint64_t(BASE_G) << shift;
        break;
      case 'T': case 't':
        bitbase = uint64_t(BASE_T) << shift;
        break;
      default:
        throw std::runtime_error("invalid DNA base");
    }
    kmers[i/32] |= bitbase;
  }
}

inline uint64_t word_reverse_complement(const uint64_t kmer, uint8_t kmer_size) {
  uint64_t res = ~kmer;
  res = ((res >> 2 & 0x3333333333333333) | (res & 0x3333333333333333) << 2);
  res = ((res >> 4 & 0x0F0F0F0F0F0F0F0F) | (res & 0x0F0F0F0F0F0F0F0F) << 4);
  res = ((res >> 8 & 0x00FF00FF00FF00FF) | (res & 0x00FF00FF00FF00FF) << 8);
  res = ((res >> 16 & 0x0000FFFF0000FFFF) | (res & 0x0000FFFF0000FFFF) << 16);
  res = ((res >> 32 & 0x00000000FFFFFFFF) | (res & 0x00000000FFFFFFFF) << 32);
  return (res >> (2 * (32 - kmer_size)));
}

void get_revcomps(const uint64_t *kmers, uint64_t *revcomps, const uint32_t size, const uint8_t kmer_size) {
  for (int i = 0; i < size; i++) {
    revcomps[i] = word_reverse_complement(kmers[i], kmer_size);
  }
}
