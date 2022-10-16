#include <inttypes.h>
#include <unordered_set>
#include <stdio.h>

#include "revcomp.h"

inline uint64_t word_reverse_complement(const uint64_t kmer, uint8_t kmerSize) {
    uint64_t res = ~kmer;
    res = ((res >> 2 & 0x3333333333333333) | (res & 0x3333333333333333) << 2);
    res = ((res >> 4 & 0x0F0F0F0F0F0F0F0F) | (res & 0x0F0F0F0F0F0F0F0F) << 4);
    res = ((res >> 8 & 0x00FF00FF00FF00FF) | (res & 0x00FF00FF00FF00FF) << 8);
    res = ((res >> 16 & 0x0000FFFF0000FFFF) | (res & 0x0000FFFF0000FFFF) << 16);
    res = ((res >> 32 & 0x00000000FFFFFFFF) | (res & 0x00000000FFFFFFFF) << 32);
    return (res >> (2 * (32 - kmerSize)));
}

void find_unique_complements(const uint64_t *kmers, const uint32_t size, 
    uint64_t **unique_complements, uint32_t *unique_complements_size) {

  std::unordered_set<uint64_t> uniques;
  
  uint32_t continues = 0;
  for (uint32_t i = 0; i < size; i++) {
    printf("%d\r", i);

    uint64_t kmer = kmers[i];
    uint64_t revcomp = word_reverse_complement(kmer, 31);

    if (uniques.find(revcomp) != uniques.end()) {
      continues++;
      continue;
    }

    uniques.insert(kmer);
  }
  printf("%d\n", size);
  printf("continues=%d\n", continues);

  *unique_complements_size = uniques.size();
  *unique_complements = new uint64_t[*unique_complements_size];

  uint32_t i = 0;
  for (const auto &kmer: uniques) {
    (*unique_complements)[i] = kmer;
    i++;
    printf("%d\r", i);
  }
  printf("%d\n", i);
}
