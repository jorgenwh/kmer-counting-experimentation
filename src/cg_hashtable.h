#ifndef CG_HASHTABLE_H_
#define CG_HASHTABLE_H_

#include <iostream>
#include <sstream>
#include <inttypes.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "cg_kernels.h"

class CGHashTable {
public:
  CGHashTable() = default;
  CGHashTable(const uint64_t *keys, const bool cuda_keys, const uint32_t size, const uint32_t capacity);
  ~CGHashTable() { 
    cudaFree(table_m.keys); 
    cudaFree(table_m.values); 
  }

  uint32_t size() const { return size_m; }
  uint32_t capacity() const { return capacity_m; }

  void count(const uint64_t *keys, const uint32_t size);
  void countcu(const uint64_t *keys, const uint32_t size);

  void get(const uint64_t *keys, uint32_t *counts, uint32_t size) const;
  void getcu(const uint64_t *keys, uint32_t *counts, uint32_t size) const;

  std::string to_string() const;
private:
  uint32_t size_m;
  uint32_t capacity_m;
  Table table_m;

  void initialize(const uint64_t *keys, const bool cuda_keys, const uint32_t size, const uint32_t capacity);
};

#endif // CG_HASHTABLE_H_
