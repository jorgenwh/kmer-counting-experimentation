#include <inttypes.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "cu_common.h"
#include "kernels.h"

__device__ inline uint64_t word_reverse_complement(const uint64_t kmer, uint8_t kmer_size) {
  uint64_t res = ~kmer;
  res = ((res >> 2 & 0x3333333333333333) | (res & 0x3333333333333333) << 2);
  res = ((res >> 4 & 0x0F0F0F0F0F0F0F0F) | (res & 0x0F0F0F0F0F0F0F0F) << 4);
  res = ((res >> 8 & 0x00FF00FF00FF00FF) | (res & 0x00FF00FF00FF00FF) << 8);
  res = ((res >> 16 & 0x0000FFFF0000FFFF) | (res & 0x0000FFFF0000FFFF) << 16);
  res = ((res >> 32 & 0x00000000FFFFFFFF) | (res & 0x00000000FFFFFFFF) << 32);
  return (res >> (2 * (32 - kmer_size)));
}

__global__ void init_hashtable_kernel(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < size) {
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    while (true) {
      unsigned long long int *old_ptr = reinterpret_cast<unsigned long long int *>(&table.keys[hash]);
      uint64_t old = atomicCAS(old_ptr, kEmpty, key);

      if (old == kEmpty || old == key) {
        table.values[hash] = 0;
        return;
      }
      hash = (hash + 1) % capacity;
    }
  }
}

void init_hashtable(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity) {
  int min_grid_size;
  int thread_block_size = 512;
  /*cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      init_hashtable_kernel, 0, 0));*/

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  init_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, size, capacity);
  //cuda_errchk(cudaDeviceSynchronize());
}

__global__ void lookup_hashtable_kernel(Table table, 
    const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < size) {
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    while (true) {
      uint64_t cur_key = table.keys[hash];
      if (cur_key == key || cur_key == kEmpty) {
        counts[thread_id] = (cur_key == key) ? table.values[hash] : 0;
        return;
      }
      hash = (hash + 1) % capacity;
    }
  }
}

void lookup_hashtable(Table table, 
    const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity) {
  int min_grid_size;
  int thread_block_size = 512;
  /*cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      lookup_hashtable_kernel, 0, 0));*/

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  lookup_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, counts, size, capacity);
  //cuda_errchk(cudaDeviceSynchronize());
}

__global__ void count_hashtable_kernel(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < size) {
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    //printf("%llu -> %llu\n", key, revcomp);

    // Search for original key
    while (true) {
      uint64_t cur_key = table.keys[hash];
      if (cur_key == kEmpty) { break; }
      if (cur_key == key) {
        atomicAdd((unsigned int *)&(table.values[hash]), 1);
        return;
      }

      hash = (hash + 1) % capacity;
    }

    key = word_reverse_complement(key, 31);
    hash = key % capacity;

    // Search for reverse complement of key
    while (true) {
      uint64_t cur_key = table.keys[hash];
      if (cur_key == kEmpty) { return; }
      if (cur_key == key) {
        atomicAdd((unsigned int *)&(table.values[hash]), 1);
        return;
      }

      hash = (hash + 1) % capacity;
    }
  }
}

void count_hashtable(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity) {
  int min_grid_size;
  int thread_block_size = 512;
  /*cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      count_hashtable_kernel, 0, 0));*/

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  count_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, size, capacity);
  //cuda_errchk(cudaDeviceSynchronize());
}

