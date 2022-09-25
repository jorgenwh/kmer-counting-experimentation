#ifndef HASHTABLE_H_
#define HASHTABLE_H_

#include <iostream>
#include <sstream>
#include <inttypes.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"

class CuHashTable {
public:
  CuHashTable() = default;
  CuHashTable(const uint64_t *keys, const uint32_t num_keys);
  ~CuHashTable() { cudaFree(table_m); }

  uint32_t size() const { return size_m; }
  uint32_t capacity() const { return HASH_TABLE_CAP; }
  uint32_t empty_slots() const;
  Values get(const uint64_t *keys, const uint32_t num_keys) const;
  void count(const uint64_t *keys, const uint32_t num_keys);
  void countcu(const uint64_t *keys, const uint32_t num_keys);

  std::string to_string() const;

  void get_keys_and_values(uint64_t **keys_ptr, uint64_t **values_ptr, uint32_t &num_keys, uint32_t &num_values) const;

private:
  uint32_t size_m;
  KeyValue *table_m;

  void initialize(const uint64_t *keys, const uint32_t num_keys);
};

#endif // HASHTABLE_H_
