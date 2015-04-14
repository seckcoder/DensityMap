
#include <iostream>
#include <cassert>
#include "config.h"
#include "cuda_util.h"
using std::cout;
using std::endl;

#undef MAX_SHARED_MEM
// #define MAX_SHARED_MEM


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



static
void kde2DParallelObject(
    float **objCoords,
    int numObjs,  // 10 - 100000
    int width,  // 1024
    int height, // 768
    float sigma,
    float **densityMap
    ) {
  
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



static
void kde2DParallelMap(
    float **objCoords,
    int numObjs,
    int width,
    int height,
    float sigma,
    float **densityMap
    ) {

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

/*
 * Using fitted shared memory. That is,
 * use the same size of shared memory
 * as block size.
 * 
 * In theory, if we increase number of
 * threads for each block, there should
 * be speedup since number of read
 * from global memory(deviceObjs) is
 * reduced.
 * But in pracice, we get max performance
 * for 512 number of threads per block.
 * And the speed up from 128 to 512 is not
 * apparent.
 *
 */

__global__ static
void sharedFitMemParallelMap(
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
    // copy the ith block of mem from deviceObjs to
    // shared memory
    int coordIdx = i * blockDim.x + threadIdx.x;
    sharedMem[2*threadIdx.x] = deviceObjs[coordIdx * 2];
    sharedMem[2*threadIdx.x+1] = deviceObjs[coordIdx * 2+1];
    __syncthreads();
    // computation for the ith block
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

/*
 * sharedMaxMemParallelMap use a large trunk of
 * shared Memory and load the shared Memory
 * for each block of threads.
 *
 * In theory, this should not be faster than
 * the fitted version since it won't reduce
 * data read from global memory.
 * In practice, we find it to be 4 times slower
 * than the fitted version.
 * This version can still be optimized since
 * the read from global memory and write
 * to shared memory is not coalesced.
 */
__global__ static
void sharedMaxMemParallelMap(
    float *deviceObjs,
    int numObjs,
    int width,
    int height,
    float sigma,
    const int numBlocksPerSharedBlock,
    float *deviceDensityMap) {
  extern __shared__ float sharedMem[];
  const int mapCoordIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const int mapCoordI = mapCoordIdx / height;
  const int mapCoordJ = mapCoordIdx % height;
  const float mapCoordX = float(mapCoordI) / float(width-1);
  const float mapCoordY = float(mapCoordJ) / float(height-1);
  float estimate = 0.f;
  const int numObjsPerBlock = blockDim.x;
  const int numObjsPerSharedBlock = numBlocksPerSharedBlock * numObjsPerBlock;
  const int numSharedBlocks = numObjs / numObjsPerSharedBlock;
  int numObjsInLastSharedBlock = numObjs % numObjsPerSharedBlock;
  const int numBlocksInLastSharedBlock = numObjsInLastSharedBlock / numObjsPerBlock;
  const int numObjsInLastBlock = numObjsInLastSharedBlock % numObjsPerBlock;

/* Unprotected macro, use it carefully.
 * The meaning of following macro is to make writing and reading
 * he following code easier by spliting the logic into
 * several independent parts.
 * I intentionally omit the argument, which makes it a little
 * confusing but more maintainable since it will report errors
 * if used in the wrong place.
 * I don't use inline method here since it's cubersome to pass
 * all these arguments around.
 */

#define loadObject() {\
    sharedMem[2*sharedObjectIdx] = deviceObjs[2*objectIdx];\
    sharedMem[2*sharedObjectIdx+1] = deviceObjs[2*objectIdx+1];\
  }
#define loadBlock() {\
    int sharedObjectIdx = blkIdx * numObjsPerBlock + threadIdx.x;\
    int objectIdx = sharedBlockIdx * numObjsPerSharedBlock +\
      sharedObjectIdx;\
    loadObject();\
  }
#define loadSharedBlock() {\
    for (int blkIdx = 0; blkIdx < numBlocksPerSharedBlock; blkIdx+=1) {\
      loadBlock();\
    }\
  }
#define reduceSharedMem(size) {\
  for (int __i__ = 0; __i__ < (size); __i__ += 1) {\
    estimate += gauss2d(sharedMem[2*__i__], sharedMem[2*__i__+1],\
        sigma, mapCoordX, mapCoordY);\
  }\
}

  for (int sharedBlockIdx = 0;
      sharedBlockIdx < numSharedBlocks;
      sharedBlockIdx++) {
    loadSharedBlock();
    __syncthreads();
    reduceSharedMem(numObjsPerSharedBlock);
  }

  int lastSharedBlockIdx = numSharedBlocks;
  for (int blkIdx = 0; blkIdx < numBlocksInLastSharedBlock; blkIdx++) {
    int sharedBlockIdx = lastSharedBlockIdx;
    loadBlock();
  }
  int lastBlkIdx = numBlocksInLastSharedBlock;
  for (int blkObjIdx = 0; blkObjIdx < numObjsInLastBlock; blkObjIdx++) {
    int sharedObjectIdx = lastBlkIdx * numObjsPerBlock + threadIdx.x;
    int objectIdx = lastSharedBlockIdx * numObjsPerSharedBlock +
      sharedObjectIdx;
    loadObject();
  }
  __syncthreads();
  reduceSharedMem(numObjsInLastSharedBlock);
  if (mapCoordIdx < width * height) {
    deviceDensityMap[mapCoordIdx] = estimate / numObjs;
  }
}

static
void kde2DParallelMapSharedMem(
    float **objCoords,
    int numObjs,
    int width,
    int height,
    float sigma,
    float **densityMap
    ) {

  const int numThreadsPerBlock = 512;
  const int numBlocks = ceilDivide(width * height,numThreadsPerBlock);
  float *deviceDensityMap;
  cudaMalloc(&deviceDensityMap, width * height * sizeof(float));
  float *deviceObjs;
  cudaMalloc(&deviceObjs, numObjs * 2 * sizeof(float));
  cudaMemcpy(deviceObjs, objCoords[0], numObjs * 2 * sizeof(float), cudaMemcpyHostToDevice);



#ifdef DEBUG
  cout << "numThreadsPerBlock: " << numThreadsPerBlock << "\n"
       << "numBlocks: " << numBlocks << endl;
#endif

#ifdef MAX_SHARED_MEM
  msg("Maximize shared memory used\n");
  /*
   * TODO: modify this
   */
  const int numBlocksPerSharedBlock = 40;
  const size_t sharedMemSize = numBlocksPerSharedBlock * numThreadsPerBlock * 2 * sizeof(float);
  sharedMaxMemParallelMap<<<numBlocks, numThreadsPerBlock,
      sharedMemSize>>>(
      deviceObjs,
      numObjs,
      width,
      height,
      sigma,
      numBlocksPerSharedBlock,
      deviceDensityMap);
#else
  const size_t sharedMemSize = numThreadsPerBlock * 2 * sizeof(float);
  sharedFitMemParallelMap<<<numBlocks, numThreadsPerBlock, sharedMemSize>>>(
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


__global__ static
void sharedMemParallelObject(
    float *deviceObjs,
    int numObjs,
    int width,
    int height,
    float sigma,
    float *deviceDensityMap) {
  extern __shared__ float sharedMem[];

  int objectIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (objectIdx < numObjs) {
    sharedMem[threadIdx.x * 2] = deviceObjs[objectIdx * 2];
    sharedMem[threadIdx.x * 2+1] = deviceObjs[objectIdx * 2+1];
    __syncthreads();
    for (int mapCoordIdx = 0; mapCoordIdx < width * height;
        mapCoordIdx += 1) {
      int mapCoordI = mapCoordIdx / height;
      int mapCoordJ = mapCoordIdx % height;
      float mapCoordX = float(mapCoordI) / float(width-1);
      float mapCoordY = float(mapCoordJ) / float(height-1);
      deviceDensityMap[mapCoordIdx] +=
        gauss2d(
            sharedMem[2*threadIdx.x],
            sharedMem[2*threadIdx.x+1],
            sigma,
            mapCoordX,
            mapCoordY);
    }
  }
}

static
void kde2DParallelObjectSharedMem(
    float **objCoords,
    int numObjs,
    int width,
    int height,
    float sigma,
    float **densityMap) {
  const int numThreadsPerBlock = 128;
  const int numBlocks = ceilDivide(numObjs,
                                   numThreadsPerBlock);
  const int sharedMemSize = numThreadsPerBlock * 2 * sizeof(float);
  float *deviceDensityMap;
  cudaMalloc(&deviceDensityMap, width * height * sizeof(float));
  cudaMemset(deviceDensityMap, 0, width * height * sizeof(float));
  float *deviceObjs;
  cudaMalloc(&deviceObjs, numObjs * 2 * sizeof(float));
  cudaMemcpy(deviceObjs, objCoords[0], numObjs * 2 * sizeof(float), cudaMemcpyHostToDevice);

#ifdef DEBUG
  cout << "numThreadsPerBlock: " << numThreadsPerBlock << "\n"
       << "numBlocks: " << numBlocks << endl;
#endif

  sharedMemParallelObject<<<
      numBlocks,
      numThreadsPerBlock,
      sharedMemSize>>>(
          deviceObjs,
          numObjs,
          width,
          height,
          sigma,
          deviceDensityMap);
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

  if (parallel_method == PARALLEL_MAP_SHARED_MEM) {
#ifdef DEBUG
    cout << "Parallel Map Update with shared memory for"
        " objectCoords" << endl;
#endif
    kde2DParallelMapSharedMem(
        objCoords,
        numObjs,
        width,
        height,
        sigma,
        densityMap
        );
  } else if (parallel_method == PARALLEL_OBJECT_SHARED_MEM) {
    kde2DParallelObjectSharedMem(
        objCoords,
        numObjs,
        width,
        height,
        sigma,
        densityMap);
  } else if ((parallel_method == PARALLEL_AUTO && width * height > numObjs) ||
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
  } else if ((parallel_method == PARALLEL_AUTO && width * height < numObjs) ||
             (parallel_method == PARALLEL_OBJECT)) {
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
  } else {
    err("UNKNOWN METHOD: %d", parallel_method);
  }
}
