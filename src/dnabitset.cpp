#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>

#include "dnabitset.h"

DNABitset::DNABitset(const char *dna_string, const size_t dna_length) {
  m_length = dna_length;
  m_bytes = (dna_length / 4) + (dna_length % 4 != 0);

  m_bitset = new uint8_t[m_bytes];
  memset(m_bitset, 0, m_bytes);

  for (size_t i = 0; i < dna_length; i++) {
    uint8_t shift = 6 - 2*(i%4);

    switch (dna_string[i]) {
      case 'A': case 'a':
        m_bitset[i/4] |= BASE_A << shift;
        break;
      case 'C': case 'c':
        m_bitset[i/4] |= BASE_C << shift;
        break;
      case 'G': case 'g':
        m_bitset[i/4] |= BASE_G << shift;
        break;
      case 'T': case 't':
        m_bitset[i/4] |= BASE_T << shift;
        break;
      default:
        assert(0);
    }
    shift = (shift == 0) ? 6 : shift - 2;
  }
}

char *DNABitset::to_string() const {
  char *dna_string = new char[m_length + 1];

  for (size_t i = 0; i < m_length; i++) {
    uint8_t shift = 6 - 2*(i%4);
    uint8_t mask = BASE_MASK << shift;

    // Get the i-th base
    uint8_t base = (m_bitset[i/4] & mask) >> shift;
    switch (base) {
      case BASE_A:
        dna_string[i] = 'a';
        break;
      case BASE_C:
        dna_string[i] = 'c';
        break;
      case BASE_G:
        dna_string[i] = 'g';
        break;
      case BASE_T:
        dna_string[i] = 't';
        break;
      default:
        assert(0);
    }
  }
  dna_string[m_length] = '\0';
  return dna_string;
}
