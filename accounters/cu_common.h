#ifndef CU_COMMON_H_
#define CU_COMMON_H_

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define _ERR_CHECK 

#define cuda_errchk(err) { cuda_errcheck(err, __FILE__, __LINE__); }
inline void cuda_errcheck(cudaError_t code, const char *file, int line, bool abort=true) {
#ifdef _ERR_CHECK
  if (code != cudaSuccess) {
    switch (code) {
      case 2:
        fprintf(stderr, "CUDA out of memory error in %s at line %d\n", file, line);
        break;
      default:
        fprintf(stderr, "CUDA assert: '%s', in %s, at line %d\n", cudaGetErrorString(code), file, line);
    }
    exit(code);
  }
#endif // _ERR_CHECK
}

#endif // CU_COMMON_H_
