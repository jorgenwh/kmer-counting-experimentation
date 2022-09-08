/*
  Source used for this: https://diego.assencio.com/?index=79a3928625303f53593f2112ebd8ac86
*/

#ifndef DNABITSET_H_
#define DNABITSET_H_

#define BASE_MASK 0x3 // Binary 11

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>

enum {
  BASE_A = 0x0, // Binary 00
  BASE_C = 0x1, // Binary 01
  BASE_G = 0x2, // Binary 10
  BASE_T = 0x3, // Binary 11
};

class DNABitset {
public:
  DNABitset(const char *dna_string, const size_t dna_length);
  ~DNABitset() { delete[] m_bitset; }

  char *to_string() const;
  size_t bytes() const { return m_bytes; }
  size_t length() const { return m_length; }
private:
  uint8_t *m_bitset;
  size_t m_length;
  size_t m_bytes;
};

#endif // DNABITSET_H_
