
#include <iostream>
#include <cassert>
#include "config.h"
using std::cout;
using std::endl;

static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints

    return ++n;
}

template <class T>
inline T getFirstDeviceValue(T *device_arr) {
  T ans;
  cudaMemcpy(&ans, device_arr, sizeof(T), cudaMemcpyDeviceToHost);
  return ans;
}

__device__ inline static
float gauss2d(float center_x, float center_y, float sigma, float x, float y) {
  float sigma_square = sigma * sigma;
  float sigma_square_inv = 1.f / sigma_square;
  float delta_x = x - center_x,
        delta_y = y - center_y;

  return 1.f/(2.f*PI*sigma_square) * exp(-0.5f * sigma_square_inv * (delta_x*delta_x + delta_y * delta_y));
}


// for debugging purpose
#ifdef DEBUG
__global__ static
void estimateCoordSeq(
    float *objCoords,
    int numObjs,
    float center_x,
    float center_y,
    float sigma,
    float *estimate_block_acc) {
  float estimate = 0.f;
  for (int i = 0; i < numObjs; i++) {
    estimate += gauss2d(center_x,center_y,sigma,objCoords[i*2],objCoords[i*2+1]);
  }
  estimate_block_acc[0] = estimate;
}

__global__ static
void reduceSeq(
    float *array,
    int n) {
  float res;
  for (int i = 0; i < n; i++) {
    res += array[i];
  }
  array[0] = res;
}

void printIntermediates(float *deviceIntermediates, float *intermediates, int n) {
  cudaMemcpy(intermediates, deviceIntermediates, n * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int k = 0; k < n; k++) {
    cout << intermediates[k] << " ";
  }
  cout << endl;
}
#endif

__global__ static
void estimateCoord(
    float *objCoords,
    int numObjs,
    float center_x,
    float center_y,
    float sigma,
    float *estimates_block_acc) {
  extern __shared__ float gauss_estimates[];
  int objectId = blockDim.x * blockIdx.x + threadIdx.x;
  // initialize shared memory
  gauss_estimates[threadIdx.x] = 0.f;
  if (objectId < numObjs) {
    float x = objCoords[objectId*2],
          y = objCoords[objectId*2+1];
    gauss_estimates[threadIdx.x] = gauss2d(center_x, center_y, sigma, x, y);
  }
  __syncthreads();
  // reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      gauss_estimates[threadIdx.x] += gauss_estimates[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    estimates_block_acc[blockIdx.x] = gauss_estimates[0];
  }
}

// reduce the array to array[0]
__global__
void reduce(float *array) {
  extern __shared__ float cache[];
  cache[threadIdx.x] = array[threadIdx.x];
  __syncthreads();
  
  // reduce
  
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      cache[threadIdx.x] += cache[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    array[0] = cache[0];
  }
}

void kde2D(
    float **objCoords,
    int numObjs,  // 10 - 100000
    float **densityMap,
    int width,  // 1024
    int height, // 768
    float sigma) {
  
  float *deviceObjs;
  cudaMalloc(&deviceObjs, numObjs * 2 * sizeof(float));
  cudaMemcpy(deviceObjs, objCoords[0], numObjs * 2 * sizeof(float), cudaMemcpyHostToDevice);

  const int numThreadsPerBlock = 1024;
  const int numBlocks = int(std::ceil((float)numObjs / (float)numThreadsPerBlock));
  const int clusterBlockSharedDataSize = numThreadsPerBlock * sizeof(float);

  // for reduction
  const int numReductionThreads = nextPowerOfTwo(numBlocks);
  const int reductionSharedDataSize = numReductionThreads * sizeof(float);

  assert(numReductionThreads <= 1024);

  float *deviceIntermediates;
  cudaMalloc(&deviceIntermediates, numReductionThreads * sizeof(float));
  cudaMemset(deviceIntermediates, 0, numReductionThreads * sizeof(float));

  float *intermediates = (float *)malloc(numReductionThreads * sizeof(float));
  
  
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      float x = float(i) / float(width - 1);
      float y = float(j) / float(height - 1);
      // estimateCoordSeq<<<1,1>>>(deviceObjs, numObjs, x,y,sigma, deviceIntermediates);
      estimateCoord<<<numBlocks, numThreadsPerBlock,
        clusterBlockSharedDataSize>>>(
          deviceObjs,
          numObjs,
          x,y, // center
          sigma, // sigma coeff
          deviceIntermediates
          );
      // reduceSeq<<<1,1>>>(deviceIntermediates, numBlocks);
      reduce<<<1, numReductionThreads, reductionSharedDataSize>>>(
          deviceIntermediates);
      densityMap[i][j] = getFirstDeviceValue(deviceIntermediates);
    }
  }
  cudaFree(deviceObjs);
  cudaFree(deviceIntermediates);
}
