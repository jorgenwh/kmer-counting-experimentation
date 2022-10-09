#ifndef COPS_KERNELS_H_
#define COPS_KERNELS_H_

#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"

namespace cops_kernels {

void init_hashtable(Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity, const uint32_t cg_size);
void count_hashtable(Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity, const uint32_t cg_size);
void lookup_hashtable(Table table, const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity, const uint32_t cg_size); 

} // cops_kernels

#endif // COPS_KERNELS_H_
