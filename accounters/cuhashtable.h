#ifndef CUHASHTABLE_H_
#define CUHASHTABLE_H_

#include <sstream>
#include <inttypes.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "cu_common.h"
#include "kernels.h"

class CuHashTable {
public:
  CuHashTable() = default;
  CuHashTable(const uint64_t *keys, const bool cuda_keys, const uint32_t size, const uint32_t capacity);
  ~CuHashTable() { 
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

#endif // CUHASHTABLE_H_
