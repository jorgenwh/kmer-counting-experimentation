#ifndef NAIVE_HASHTABLE_H_
#define NAIVE_HASHTABLE_H_

#include <iostream>
#include <sstream>
#include <inttypes.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"

class NaiveHashTable {
public:
  NaiveHashTable() = default;
  NaiveHashTable(const uint64_t *keys, const uint32_t size);
  ~NaiveHashTable() { cudaFree(table_m); }

  uint32_t size() const { return size_m; }
  uint32_t capacity() const { return HASH_TABLE_CAP; }

  void count(const uint64_t *keys, const uint32_t size);
  void countcu(const uint64_t *keys, const uint32_t size);

  void get(const uint64_t *keys, uint64_t *counts, uint32_t size) const;
  void getcu(const uint64_t *keys, uint64_t *counts, uint32_t size) const;

  std::string to_string() const;

private:
  uint32_t size_m;
  KeyValue *table_m;

  void initialize(const uint64_t *keys, const uint32_t size);
};

#endif // NAIVE_HASHTABLE_H_
