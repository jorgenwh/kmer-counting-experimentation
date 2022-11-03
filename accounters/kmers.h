#ifndef KMERS_H_
#define KMERS_H_

#include <stdexcept>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <unordered_set>

#define ASCII_A 65
#define ASCII_C 67
#define ASCII_G 71
#define ASCII_T 84

#define BASE_MASK 0x3 // Binary 11

enum {
  BASE_A = 0x0, // Binary 00
  BASE_C = 0x1, // Binary 01
  BASE_G = 0x2, // Binary 10
  BASE_T = 0x3, // Binary 11
};

extern uint8_t shifts8b[4];
extern uint8_t shifts64b[32];

void hashes_to_ascii(const uint64_t *hashes, const uint64_t num_hashes, uint8_t *bases, const uint64_t num_bases);
void encode_kmers(const char *bases, const uint32_t num_kmers, uint64_t *kmers, const uint32_t kmer_size);
void get_reverse_complements(const uint64_t *kmers, uint64_t *revcomps, const uint32_t size, const uint8_t kmer_size);
void convert_ACTG_to_ACGT_encoding(const uint64_t *kmers, uint64_t *ret, const uint32_t size);
std::unordered_set<uint64_t> get_unique_complements_set(const uint64_t *kmers, const uint32_t size);
#endif // KMERS_H_
