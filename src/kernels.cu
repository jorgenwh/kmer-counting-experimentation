#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"

__global__ void init_hashtable_kernel(
    KeyValue *table, const uint64_t *keys, const uint32_t num_keys, const uint32_t capacity) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < num_keys) {
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    while (true) {
      unsigned long long int *old_ptr = reinterpret_cast<unsigned long long int *>(&table[hash].key);
      uint64_t old = atomicCAS(old_ptr, kEmpty, key);

      if (old == kEmpty || old == key) {
        table[hash].value = 0;
        return;
      }
      hash = (hash + 1) % capacity;
    }
  }
}

void init_hashtable(
    KeyValue *table, const uint64_t *keys, const uint32_t num_keys, const uint32_t capacity) {
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      init_hashtable_kernel, 0, 0));

  int grid_size = num_keys / thread_block_size + (num_keys % thread_block_size > 0);
  init_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, num_keys, capacity);
  cuda_errchk(cudaDeviceSynchronize());
}

__global__ void lookup_hashtable_kernel(KeyValue *table, 
    const uint64_t *keys, uint64_t *values, const uint32_t num_keys, const uint32_t capacity) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < num_keys) {
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    while (true) {
      KeyValue cur = table[hash];
      if (cur.key == key || cur.key == kEmpty) {
        values[thread_id] = (cur.key == key) ? cur.value : vInvalid;
        return;
      }
      hash = (hash + 1) % capacity;
    }
  }
}

void lookup_hashtable(KeyValue *table, const uint64_t *keys, uint64_t *values, 
    const uint32_t num_keys, const uint32_t capacity) {
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      lookup_hashtable_kernel, 0, 0));

  int grid_size = num_keys / thread_block_size + (num_keys % thread_block_size > 0);
  lookup_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, values, num_keys, capacity);
  cuda_errchk(cudaDeviceSynchronize());
}

__global__ void count_hashtable_kernel(
    KeyValue *table, const uint64_t *keys, const uint32_t num_keys, const uint32_t capacity) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < num_keys) {
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    while (true) {
      if (table[hash].key == kEmpty) { return; }
      if (table[hash].key == key) {
        atomicAdd((unsigned long long int *)&(table[hash].value), 1);
        return;
      }

      hash = (hash + 1) % capacity;
    }
  }
}

void count_hashtable(
    KeyValue *table, const uint64_t *keys, const uint32_t num_keys, const uint32_t capacity) {
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      count_hashtable_kernel, 0, 0));

  int grid_size = num_keys / thread_block_size + (num_keys % thread_block_size > 0);
  count_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, num_keys, capacity);
  cuda_errchk(cudaDeviceSynchronize());
}
