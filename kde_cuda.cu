
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

__host__ __device__ inline static
float gauss2d(float centerX, float centerY, float sigma, float x, float y) {
  float sigma_square = sigma * sigma;
  float sigma_square_inv = 1.f / sigma_square;
  float delta_x = x - centerX,
        delta_y = y - centerY;

  return 1.f/(2.f*PI*sigma_square) * exp(-0.5f * sigma_square_inv * (delta_x*delta_x + delta_y * delta_y));
}


// for debugging purpose
#ifdef DEBUG
__global__ static
void estimateCoordSeq(
    float *objCoords,
    int numObjs,
    float x,
    float y,
    float sigma,
    float *estimate_block_acc) {
  float estimate = 0.f;
  for (int i = 0; i < numObjs; i++) {
    estimate += gauss2d(objCoords[i*2],objCoords[i*2+1], sigma, x, y);
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

__global__ static
void updateDensityMapSeq(
    float centerX,
    float centerY,
    int width,
    int height,
    float sigma,
    float *deviceDensityMap) {
  for (int mapCoordIdx = 0; mapCoordIdx < 128; mapCoordIdx++) {
    int i = mapCoordIdx / height;
    int j = mapCoordIdx % height;
    float x = (float)i / (float)(width-1);
    float y = (float)j / (float)(height-1);
    if (i < width) {
      deviceDensityMap[mapCoordIdx] += gauss2d(
          centerX,
          centerY,
          sigma,
          x,y);
    }
  }
}

#endif

__global__ static
void estimateCoord(
    float *objCoords,
    int numObjs,
    float x,
    float y,
    float sigma,
    float *estimatesBlockAcc) {
  extern __shared__ float gauss_estimates[];
  int objectId = blockDim.x * blockIdx.x + threadIdx.x;
  // initialize shared memory
  gauss_estimates[threadIdx.x] = 0.f;
  if (objectId < numObjs) {
    float centerX = objCoords[objectId*2],
          centerY = objCoords[objectId*2+1];
    gauss_estimates[threadIdx.x] = gauss2d(centerX, centerY, sigma, x, y);
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
    estimatesBlockAcc[blockIdx.x] = gauss_estimates[0];
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



void kde2DParallelObject(
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
          x,y, // map coord
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


__global__ static
void updateDensityMap(
    float centerX,
    float centerY,
    int width,
    int height,
    float sigma,
    float *deviceDensityMap) {
  int mapCoordIdx = blockDim.x * blockIdx.x + threadIdx.x;

  int i = mapCoordIdx / height;
  int j = mapCoordIdx % height;
  if (i < width) {
    float x = float(i) / float(width-1);
    float y = float(j) / float(height-1);
    deviceDensityMap[mapCoordIdx] += gauss2d(
        centerX, /* center */
        centerY, 
        sigma,
        x,y);
  }
}

void kde2DParallelMap(
    float **objCoords,
    int numObjs,
    float **densityMap,
    int width,
    int height,
    float sigma) {

  float *deviceDensityMap;
  cudaMalloc(&deviceDensityMap, width * height * sizeof(float));
  cudaMemset(deviceDensityMap, 0, width * height * sizeof(float));

  const int numThreadsPerBlock = 1024;
  const int numBlocks = int(std::ceil((float)(width * height) / (float)numThreadsPerBlock));
  for (int i = 0; i < numObjs; i++) {
    /*
    updateDensityMapSeq<<<1,1>>>(
        objCoords[i][0],
        objCoords[i][1],
        width,
        height,
        sigma,
        deviceDensityMap);
        */
    updateDensityMap<<<numBlocks, numThreadsPerBlock>>>(
        objCoords[i][0],
        objCoords[i][1],
        width,
        height,
        sigma,
        deviceDensityMap);
  }
  cudaMemcpy(densityMap[0], deviceDensityMap, width*height*sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaFree(deviceDensityMap);
}

void kde2D(
    float **objCoords,
    int numObjs,  // 10 - 100000
    float **densityMap,
    int width,  // 1024
    int height, // 768
    float sigma) {
  if (width * height > numObjs) {
#ifdef DEBUG
    cout << "Parallel Map Update" << endl;
#endif
    kde2DParallelMap(
        objCoords,
        numObjs,
        densityMap,
        width,
        height,
        sigma);
  } else {
#ifdef DEBUG
    cout << "Parallel Obj Update" << endl;
#endif
    kde2DParallelObject(
        objCoords,
        numObjs,
        densityMap,
        width,
        height,
        sigma);
  }
}
