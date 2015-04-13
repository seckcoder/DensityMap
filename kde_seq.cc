#include <iostream>
#include <algorithm>
#include <cmath>
#include "config.h"

static int parallel_method = PARALLEL_AUTO;
void setParallelMethod(int method) {
  parallel_method = method;
}

float gauss2d(float center_x, float center_y, float sigma, float x, float y) {
  float sigma_square = sigma * sigma;
  float sigma_square_inv = 1.f / sigma_square;
  float delta_x = x - center_x,
        delta_y = y - center_y;

  return 1.f/(2.f*PI*sigma_square) * exp(-0.5f * sigma_square_inv * (delta_x*delta_x + delta_y * delta_y));
}

// estimate density of each coordinate in density map
float estimateCoord(
    float **objCoords,
    int numObjs,
    float sigma,
    float x,
    float y) {
  float estimate = 0.f;
  for (int i = 0; i < numObjs; i++) {
    estimate += gauss2d(
        objCoords[i][0], // center
        objCoords[i][1],
        sigma,
        x,y // density map coord
        );
  }
  return estimate;
}

// 2 dimensional kernel density estimation
void kde2D(
    float **objCoords, // numObjs * 2
    int numObjs,
    float **densityMap, // width * height
    int width,
    int height,
    float sigma // bandwidth
    ) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      float x = float(i) / float(width-1);
      float y = float(j) / float(height-1);
      densityMap[i][j] = estimateCoord(
          objCoords,
          numObjs,
          sigma,
          x,
          y);
    }
  }
}
