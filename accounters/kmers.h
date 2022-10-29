#ifndef KMERS_H_
#define KMERS_H_

#include <stdexcept>
#include <inttypes.h>
#include <string.h>
#include <assert.h>

#define ASCII_A 65
#define ASCII_C 67
#define ASCII_T 84
#define ASCII_G 71

#define BASE_MASK 0x3 // Binary 11

enum {
  BASE_A = 0x0, // Binary 00
  BASE_C = 0x1, // Binary 01
  BASE_T = 0x2, // Binary 10
  BASE_G = 0x3, // Binary 11
};

extern uint8_t shifts[32];

void hashes_to_ascii(const uint64_t *hashes, const uint64_t num_hashes, uint8_t *bases, const uint64_t num_bases);

#endif // KMERS_H_
