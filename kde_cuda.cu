
#include <iostream>
#include <cassert>
#include "config.h"
#include "kde.h"
using std::cout;
using std::endl;



static int parallel_method = PARALLEL_AUTO;
void setParallelMethod(int method) {
  parallel_method = method;
}

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

inline int ceilDivide(int a, int b) {
  return int(std::ceil((float)a / (float) b));
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

__global__ static
void sharedMemSeqKDE(
    float *deviceObjs,
    int numObjs,
    int width,
    int height,
    float sigma,
    float *deviceDensityMap) {
  extern __shared__ char mem[];
  float *sharedMem = (float *)mem;

  int mapCoordIdx = blockDim.x * blockIdx.x + threadIdx.x;
  int mapCoordI = mapCoordIdx / height;
  int mapCoordJ = mapCoordIdx % height;
  int objectIdx = threadIdx.x;
  /* Note: No matter mapCoordI < width or > width,
   * we should copy the whole block of deviceObjs.
   * Otherwise, there will be error
   */
  if (objectIdx < numObjs) {
    sharedMem[2*threadIdx.x] = deviceObjs[objectIdx * 2];
    sharedMem[2*threadIdx.x + 1] = deviceObjs[objectIdx * 2 + 1];
  }
  __syncthreads();
  if (mapCoordI < width) {
    float mapCoordX = float(mapCoordI) / float(width-1);
    float mapCoordY = float(mapCoordJ) / float(height-1);
    float estimate = 0.f;

    for (int j = 0; j < numObjs; j++) {
      estimate += gauss2d(
          // deviceObjs[2*j],
          // deviceObjs[2*j+1],
          sharedMem[2*j],
          sharedMem[2*j+1],
          sigma,
          mapCoordX,
          mapCoordY);
    }
    deviceDensityMap[mapCoordIdx] = estimate / numObjs;
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

  const int numThreadsPerBlock = 128;
  const int numBlocks = ceilDivide(numObjs, numThreadsPerBlock);
  const int clusterBlockSharedDataSize = numThreadsPerBlock * sizeof(float);

  // for reduction
  const int numReductionThreads = nextPowerOfTwo(numBlocks);
  const int reductionSharedDataSize = numReductionThreads * sizeof(float);

#ifdef DEBUG
  cout << "numThreadsPerBlock: " << numThreadsPerBlock << "\n"
       << "numBlocks: " << numBlocks << "\n"
       << "numReductionThreads: " << numReductionThreads
       << endl;
#endif
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
      densityMap[i][j] = getFirstDeviceValue(deviceIntermediates) / numObjs;
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

__global__ static
void mapNormalize(
    int width, int height,
    int numObjs,
    float *deviceDensityMap) {
  int mapCoordIdx = blockDim.x * blockIdx.x + threadIdx.x;

  int i = mapCoordIdx / height;
  if (i < width) {
    deviceDensityMap[mapCoordIdx] /= (float)numObjs;
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

  const int numThreadsPerBlock = 128;
  const int numBlocks = int(std::ceil((float)(width * height) / (float)numThreadsPerBlock));
#ifdef DEBUG
  cout << "numThreadsPerBlock: " << numThreadsPerBlock << endl;
  cout << "numBlocks: " << numBlocks << endl;
#endif
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
  mapNormalize<<<numBlocks, numThreadsPerBlock>>>(width, height, numObjs, deviceDensityMap);
  cudaMemcpy(densityMap[0], deviceDensityMap, width*height*sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaFree(deviceDensityMap);
}


__global__ static
void sharedMemParallelKDE(
    float *deviceObjs,
    int numObjs,
    int width,
    int height,
    float sigma,
    float *deviceDensityMap) {
  extern __shared__ float sharedMem[];
  int mapCoordIdx = blockDim.x * blockIdx.x + threadIdx.x;
  int mapCoordI = mapCoordIdx / height;
  int mapCoordJ = mapCoordIdx % height;
  float mapCoordX = float(mapCoordI) / float(width-1);
  float mapCoordY = float(mapCoordJ) / float(height-1);
  float estimate = 0.f;
  int numLoops = numObjs / blockDim.x;
  // assume sharedMemSize is a multiple of blockDim.x
  for (int i = 0; i < numLoops; i += 1) {
    // copy the ith block of mem from deviceObjs
    int coordIdx = i * blockDim.x + threadIdx.x;
    sharedMem[2*threadIdx.x] = deviceObjs[coordIdx * 2];
    sharedMem[2*threadIdx.x+1] = deviceObjs[coordIdx * 2+1];
    __syncthreads();
    for (int j = 0; j < blockDim.x; j++) {
      estimate += gauss2d(
          sharedMem[2*j], sharedMem[2*j+1], // center
          sigma,
          mapCoordX,
          mapCoordY);
    }
  }
  // rest
  // Extract rest here to reduce number of if check
  int objectIdx = numLoops  * blockDim.x + threadIdx.x;
  if (objectIdx < numObjs) {
    sharedMem[2*threadIdx.x] = deviceObjs[objectIdx * 2];
    sharedMem[2*threadIdx.x+1] = deviceObjs[objectIdx * 2 + 1];
  }
  __syncthreads();
  for (int j = 0; j < blockDim.x && j + numLoops * blockDim.x < numObjs; j++) {
    estimate += gauss2d(
        sharedMem[2*j],
        sharedMem[2*j+1],
        sigma,
        mapCoordX,
        mapCoordY);
  }
  /*
   * Note : we need to put this check at the end.
   * Since when mapCoordI >= width, we still need
   * to load the deviceObjs to sharedMem
   */
  if (mapCoordIdx < width * height) {
    deviceDensityMap[mapCoordIdx] = estimate / numObjs;
  }
}

static
void kde2DSharedMem1(
    float **objCoords,
    int numObjs,
    int width,
    int height,
    float sigma,
    float **densityMap
    ) {

  const int numThreadsPerBlock = 128;
  const int numBlocks = ceilDivide(width * height,numThreadsPerBlock);
  float *deviceDensityMap;
  cudaMalloc(&deviceDensityMap, width * height * sizeof(float));
  float *deviceObjs;
  cudaMalloc(&deviceObjs, numObjs * 2 * sizeof(float));
  cudaMemcpy(deviceObjs, objCoords[0], numObjs * 2 * sizeof(float), cudaMemcpyHostToDevice);
  const size_t sharedMemSize = numThreadsPerBlock * 2 * sizeof(float);

#ifdef DEBUG
  cout << "numThreadsPerBlock: " << numThreadsPerBlock << "\n"
       << "numBlocks: " << numBlocks << endl;
#endif
#if 1
  sharedMemParallelKDE<<<numBlocks, numThreadsPerBlock, sharedMemSize>>>(
      deviceObjs,
      numObjs,
      width,
      height,
      sigma,
      deviceDensityMap);
#endif
#if 0
  sharedMemSeqKDE<<<numBlocks, numThreadsPerBlock, sharedMemSize>>>(
      deviceObjs,
      numObjs,
      width,
      height,
      sigma,
      deviceDensityMap);
#endif
  cudaThreadSynchronize(); checkLastCudaError();
  cudaMemcpy(densityMap[0], deviceDensityMap,
             width * height * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(deviceDensityMap);
  cudaFree(deviceObjs);
}

void kde2D(
    float **objCoords,
    int numObjs,  // 10 - 100000
    int width,  // 1024
    int height, // 768
    float sigma,
    float **densityMap
    ) {
  kde2DSharedMem1(
      objCoords,
      numObjs,
      width,
      height,
      sigma,
      densityMap
      );
#if 0
  if ((parallel_method == PARALLEL_AUTO && width * height > numObjs) ||
      (parallel_method == PARALLEL_MAP)) {
#ifdef DEBUG
    cout << "Parallel Map Update" << endl;
#endif
    kde2DParallelMap(
        objCoords,
        numObjs,
        width,
        height,
        sigma,
        densityMap
        );
  } else {
#ifdef DEBUG
    cout << "Parallel Obj Update" << endl;
#endif
    kde2DParallelObject(
        objCoords,
        numObjs,
        width,
        height,
        sigma,
        densityMap
        );
  }
#endif
}
