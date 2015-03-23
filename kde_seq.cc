#include <algorithm>
#include <cmath>

#define PI 3.1415926f


struct Gauss {
  public:
    float ux,uy,sigma;
    Gauss(float hux, float huy, float hsigma):ux(hux),uy(huy),sigma(hsigma) {}
    float operator()(float x, float y) {
      float sigma_square = sigma * sigma;
      float sigma_square_inv = 1.f / sigma_square;
      float delta_x = x - ux;
      float delta_y = y - uy;
      return 1.f/(2.f*PI*sigma_square) * exp(-0.5f * sigma_square_inv * (delta_x*delta_x + delta_y * delta_y));
    }
};


// estimate density of each coordinate in density map
float estimateCoord(
    float **objCoords,
    int numObjs,
    float sigma,
    float x,
    float y) {
  Gauss gauss_op(x,y,sigma);
  float estimate = 0.f;
  for (int i = 0; i < numObjs; i++) {
    estimate += gauss_op(objCoords[i][0], objCoords[i][1]);
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
