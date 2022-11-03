#include <stdexcept>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <unordered_set>
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

inline uint64_t kmer_reverse_complement(const uint64_t kmer, uint8_t kmer_size) {
  uint64_t res = ~kmer;

  // Nicen't
  uint64_t mask = 1;
  if (kmer_size != 32) {
    mask = (mask << kmer_size*2) - 1;
  } else {
    mask -= 2;
  }

  res = ((res >> 2 & 0x3333333333333333) | (res & 0x3333333333333333) << 2);
  res = ((res >> 4 & 0x0F0F0F0F0F0F0F0F) | (res & 0x0F0F0F0F0F0F0F0F) << 4);
  res = ((res >> 8 & 0x00FF00FF00FF00FF) | (res & 0x00FF00FF00FF00FF) << 8);
  res = ((res >> 16 & 0x0000FFFF0000FFFF) | (res & 0x0000FFFF0000FFFF) << 16);
  res = ((res >> 32 & 0x00000000FFFFFFFF) | (res & 0x00000000FFFFFFFF) << 32);
  return (res >> (2 * (32 - kmer_size))) & mask;
}

void get_reverse_complements(const uint64_t *kmers, uint64_t *revcomps, 
    const uint32_t size, const uint8_t kmer_size) {
  for (int i = 0; i < size; i++) {
    revcomps[i] = kmer_reverse_complement(kmers[i], kmer_size);
  }
}

void convert_ACTG_to_ACGT_encoding(const uint64_t *kmers, uint64_t *ret, const uint32_t size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < 32; j++) {
      uint64_t shift = shifts64b[j];
      uint64_t mask = BASE_MASK << shift;
      uint64_t bitbase = (kmers[i] & mask) >> shift;

      switch (bitbase) {
        case BASE_G:
          bitbase = BASE_T;
          break;
        case BASE_T:
          bitbase = BASE_G;
          break;
      }

      ret[i] |= (bitbase << shift);
    }
    std::cout << i << "/" << size << "\r";
  }
  std::cout << size << "/" << size << "\n";
}

std::unordered_set<uint64_t> get_unique_complements_set(const uint64_t *kmers, const uint32_t size, const uint8_t kmer_size) {
  std::unordered_set<uint64_t> uniques;
  for (int i = 0; i < size; i++) {
    std::cout << i << "/" << size << "\r";
    uint64_t kmer = kmers[i];
    uint64_t revcomp = kmer_reverse_complement(kmer, kmer_size);
    if ((uniques.find(kmer) != uniques.end()) || (uniques.find(revcomp) != uniques.end())) {
      continue;
    }
    uniques.insert(kmer);
  }
  std::cout << size << "/" << size << "\n";
  return uniques;
}
