#ifndef REVCOMP_H_
#define REVCOMP_H_

#include <inttypes.h>
#include <unordered_set>
#include <stdio.h>

void find_unique_complements(const uint64_t *kmers, const uint32_t size, 
    uint64_t **unique_complements, uint32_t *unique_complements_size);

#endif // REVCOMP_H_
