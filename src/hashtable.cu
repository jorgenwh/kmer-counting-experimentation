#include <iostream>
#include <sstream>
#include <inttypes.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"
#include "hashtable.h"

CuHashTable::CuHashTable(const uint64_t *keys, const uint32_t num_keys) {
  initialize(keys, num_keys);
}

void CuHashTable::initialize(const uint64_t *keys, const uint32_t num_keys) {
  cuda_errchk(cudaMalloc(&table_m, sizeof(KeyValue)*HASH_TABLE_CAP));
  cuda_errchk(cudaMemset(table_m, 0xff, sizeof(KeyValue)*HASH_TABLE_CAP));
  cuda_errchk(cudaDeviceSynchronize());

  uint64_t *d_keys;
  cuda_errchk(cudaMalloc(&d_keys, sizeof(uint64_t)*num_keys));
  cuda_errchk(cudaMemcpy(d_keys, keys, sizeof(uint64_t)*num_keys, cudaMemcpyHostToDevice));

  // Synchronize because cudaMemset is asynchronous with respect to host
  cuda_errchk(cudaDeviceSynchronize());

  init_hashtable(table_m, d_keys, num_keys, HASH_TABLE_CAP);
  cuda_errchk(cudaFree(d_keys));

  size_m = num_keys;
}

Values CuHashTable::get(const uint64_t *keys, const uint32_t num_keys) const {
  uint64_t *d_keys;
  cuda_errchk(cudaMalloc(&d_keys, sizeof(uint64_t)*num_keys));
  cuda_errchk(cudaMemcpy(d_keys, keys, sizeof(uint64_t)*num_keys, cudaMemcpyHostToDevice));

  uint64_t *d_values;
  cuda_errchk(cudaMalloc(&d_values, sizeof(uint64_t)*num_keys));

  Values ret;
  ret.values = new uint64_t[num_keys];
  ret.num_values = num_keys;

  lookup_hashtable(table_m, d_keys, d_values, num_keys, HASH_TABLE_CAP);

  cuda_errchk(cudaMemcpy(ret.values, d_values, sizeof(uint64_t)*num_keys, cudaMemcpyDeviceToHost));
  cuda_errchk(cudaFree(d_keys));
  cuda_errchk(cudaFree(d_values));

  return ret;
}

void CuHashTable::count(const uint64_t *keys, const uint32_t num_keys) {
  uint64_t *d_keys;
  cuda_errchk(cudaMalloc(&d_keys, sizeof(uint64_t)*num_keys));
  cuda_errchk(cudaMemcpy(d_keys, keys, sizeof(uint64_t)*num_keys, cudaMemcpyHostToDevice));

  count_hashtable(table_m, d_keys, num_keys, HASH_TABLE_CAP);
  cuda_errchk(cudaFree(d_keys));
}

void CuHashTable::countcu(const uint64_t *keys, const uint32_t num_keys) {
  count_hashtable(table_m, keys, num_keys, HASH_TABLE_CAP);
}

uint32_t CuHashTable::empty_slots() const {
  uint32_t empty_slots = 0;
  KeyValue *h_table = new KeyValue[HASH_TABLE_CAP];
  cuda_errchk(cudaMemcpy(h_table, table_m, sizeof(KeyValue)*HASH_TABLE_CAP, cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < HASH_TABLE_CAP; i++) {
    empty_slots += (h_table[i].key == kEmpty);
  }
  delete[] h_table;
  return empty_slots;
}

std::string CuHashTable::to_string() const {
  int print_size = (HASH_TABLE_CAP < 100) ? HASH_TABLE_CAP : 100;

  KeyValue *h_table = new KeyValue[HASH_TABLE_CAP];
  cuda_errchk(cudaMemcpy(h_table, table_m, sizeof(KeyValue)*HASH_TABLE_CAP, cudaMemcpyDeviceToHost));

  std::ostringstream oss;
  std::ostringstream keys_oss;
  std::ostringstream values_oss;

  keys_oss << "[";
  values_oss << "[";
  uint32_t elements = 0;
  for (int i = 0; i < HASH_TABLE_CAP; i++) {
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
  oss << ", capacity=" << HASH_TABLE_CAP << ")";

  delete[] h_table;
  return oss.str();
}

void CuHashTable::get_keys_and_values(uint64_t **keys_ptr, uint64_t **values_ptr, 
    uint32_t &num_keys, uint32_t &num_values) const {
  KeyValue *h_table = new KeyValue[size_m];
  cuda_errchk(cudaMemcpy(h_table, table_m, sizeof(KeyValue)*size_m, cudaMemcpyDeviceToHost));

  *keys_ptr = new uint64_t[size_m];
  *values_ptr = new uint64_t[size_m];
  num_keys = size_m;
  num_values = size_m;

  for (uint32_t i = 0; i < size_m; i++) {
    (*keys_ptr)[i] = h_table[i].key;
    (*values_ptr)[i] = h_table[i].value;
  }

  delete[] h_table;
}
