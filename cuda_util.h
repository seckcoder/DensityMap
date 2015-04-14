#ifndef _H_CUDA_UTIL
#define _H_CUDA_UTIL

#include "kde.h"

inline void checkCuda(cudaError_t e) {
  if (e != cudaSuccess) {
    err("CUDA Error: %s\n", cudaGetErrorString(e));
  }
}

inline void checkLastCudaError() {
  checkCuda(cudaGetLastError());
}

#endif
