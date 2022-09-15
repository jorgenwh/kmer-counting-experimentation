#ifndef CUHASHTABLE_H_
#define CUHASHTABLE_H_

#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>

struct KeyValue {
  uint64_t key;
  uint64_t value;
};

class CuHashTable{
public:
  CudaHashTable(uint64_t *keys);
  ~CudaHashTable();

  void insert(uint64_t key, uint64_t value) {
    uint64_t slot = key % mod_m;
  }

private:
  int size;
  KeyValue *hashtable_m;
  int mod_m;

};

#endif // CUHASHTABLE_H_
