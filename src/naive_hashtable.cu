#include <iostream>
#include <sstream>
#include <inttypes.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"
#include "naive_hashtable.h"

NaiveHashTable::NaiveHashTable(const uint64_t *keys, const uint32_t size, const uint32_t capacity) {
  initialize(keys, size, capacity);
}

void NaiveHashTable::initialize(const uint64_t *keys, const uint32_t size, const uint32_t capacity) {
  capacity_m = capacity;
  size_m = size;

  cuda_errchk(cudaMalloc(&table_m, sizeof(KeyValue)*capacity));
  cuda_errchk(cudaMemset(table_m, 0xff, sizeof(KeyValue)*capacity));
  cuda_errchk(cudaDeviceSynchronize());

  uint64_t *d_keys;
  cuda_errchk(cudaMalloc(&d_keys, sizeof(uint64_t)*size));
  cuda_errchk(cudaMemcpy(d_keys, keys, sizeof(uint64_t)*size, cudaMemcpyHostToDevice));

  // Synchronize because cudaMemset is asynchronous with respect to host
  cuda_errchk(cudaDeviceSynchronize());

  init_hashtable(table_m, d_keys, size, capacity);
  cuda_errchk(cudaFree(d_keys));
}

void NaiveHashTable::get(const uint64_t *keys, uint64_t *counts, uint32_t size) const {
  uint64_t *d_keys;
  cuda_errchk(cudaMalloc(&d_keys, sizeof(uint64_t)*size));
  cuda_errchk(cudaMemcpy(d_keys, keys, sizeof(uint64_t)*size, cudaMemcpyHostToDevice));

  uint64_t *d_counts;
  cuda_errchk(cudaMalloc(&d_counts, sizeof(uint64_t)*size));

  lookup_hashtable(table_m, d_keys, d_counts, size, capacity_m); 

  cuda_errchk(cudaMemcpy(counts, d_counts, sizeof(uint64_t)*size, cudaMemcpyDeviceToHost));
  cuda_errchk(cudaFree(d_keys));
  cuda_errchk(cudaFree(d_counts));
}

void NaiveHashTable::getcu(const uint64_t *keys, uint64_t *counts, uint32_t size) const {
  lookup_hashtable(table_m, keys, counts, size, capacity_m); 
}

void NaiveHashTable::count(const uint64_t *keys, const uint32_t size) {
  uint64_t *d_keys;
  cuda_errchk(cudaMalloc(&d_keys, sizeof(uint64_t)*size));
  cuda_errchk(cudaMemcpy(d_keys, keys, sizeof(uint64_t)*size, cudaMemcpyHostToDevice));

  count_hashtable(table_m, d_keys, size, capacity_m);
  cuda_errchk(cudaFree(d_keys));
}

void NaiveHashTable::countcu(const uint64_t *keys, const uint32_t size) {
  count_hashtable(table_m, keys, size, capacity_m);
}

std::string NaiveHashTable::to_string() const {
  int print_size = (capacity_m < 100) ? capacity_m : 100;

  KeyValue *h_table = new KeyValue[capacity_m];
  cuda_errchk(cudaMemcpy(h_table, table_m, sizeof(KeyValue)*capacity_m, cudaMemcpyDeviceToHost));

  std::ostringstream oss;
  std::ostringstream keys_oss;
  std::ostringstream values_oss;

  keys_oss << "[";
  values_oss << "[";
  uint32_t elements = 0;
  for (int i = 0; i < capacity_m; i++) {
    KeyValue cur = h_table[i];
    if (cur.key == kEmpty) { continue; }

    if (elements != 0) { 
      keys_oss << ", "; 
      values_oss << ", "; 
    }

    keys_oss << cur.key;
    values_oss << cur.value;
    
    elements++;
    if (elements >= print_size) { break; }
  }
  keys_oss << "]";
  values_oss << "]";

  oss << "Counter(" << keys_oss.str() << ", " << values_oss.str();
  oss << ", capacity=" << capacity_m << ")";

  delete[] h_table;
  return oss.str();
}
