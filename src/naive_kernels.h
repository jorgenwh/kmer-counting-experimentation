#ifndef NAIVE_KERNELS_H_
#define NAIVE_KERNELS_H_

#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"

namespace naive_kernels {

void init_hashtable(Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity);
void count_hashtable(Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity);
void lookup_hashtable(Table table, const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity); 

} // naive_kernels

#endif // NAIVE_KERNELS_H_
