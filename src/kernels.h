#ifndef KERNELS_H_
#define KERNELS_H_

#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"

void init_hashtable(KeyValue *table, const uint64_t *keys, uint32_t num_keys, uint32_t capacity);
void lookup_hashtable(KeyValue *table, const uint64_t *keys, uint64_t *values, const uint32_t num_keys, const uint32_t capacity);
void count_hashtable(KeyValue *table, const uint64_t *keys, const uint32_t num_keys, const uint32_t capacity);

#endif // KERNELS_H_
