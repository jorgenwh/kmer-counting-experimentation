#ifndef CU_H_
#define CU_H_

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define _DEBUG

#define HASH_TABLE_CAP 53070300
//#define HASH_TABLE_CAP 250000000

static const uint64_t kEmpty = 0xffffffffffffffff;
static const uint64_t vInvalid = 0xffffffffffffffff;

#define cuda_errchk(err) { cuda_errcheck(err, __FILE__, __LINE__); }
inline void cuda_errcheck(cudaError_t code, const char *file, int line, bool abort=true) {
#ifdef _DEBUG
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA assert: '%s', in %s, at line %d\n", cudaGetErrorString(code), file, line);
    if (abort) { exit(code); }
  }
#endif // _DEBUG
}

struct KeyValue {
  uint64_t key;
  uint64_t value;
};

struct Values {
  uint64_t *values;
  uint32_t num_values;
};

#endif // CU_H_
