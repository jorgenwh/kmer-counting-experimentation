#include <stdexcept>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <iostream>

#include "kmers.h"

uint8_t shifts[32] = {
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
    uint8_t shift = shifts[i%32];
    uint8_t mask = BASE_MASK << shift;

    uint8_t bitbase = (hashes[i/32] & mask) >> shift;
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

