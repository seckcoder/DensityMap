#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include "kde.h"
#include "config.h"

using namespace std;

static void usage(char *argv0) {
  const char *help =
      "Usage: %s [switches] -i filename\n"
      "       -i filename               : file containing 2D data used to generate density map\n"
      "       -s sigma[default=0.1]     : bandwidth for kernel density estimation\n"
      "       -w width[default=800]     : width of density map\n"
      "       -h height[default=800]    : height of density map\n"
      "       -p parallel[default=0]    : Parallel Method. AUTO(%d), MAP(1), OBJECT(2)\n"
      "       -o output                 : output file name of density map\n";
  fprintf(stderr, help, argv0);
  exit(-1);
}

int main(int argc, char **argv) {
  int opt;
  char *input_fname;
  char *output_fname;
  float sigma = 0.1;
  double timing, io_timing, kde_timing;
  int width = 800, height = 800;
  int parallel_method = PARALLEL_AUTO;
  while ( (opt=getopt(argc,argv,"i:s:w:h:p:o:") ) != EOF ) {
    switch (opt) {
      case 'i':
        input_fname=optarg;
        break;
      case 's':
        sigma = atof(optarg);
        break;
      case 'w':
        width = atoi(optarg);
        break;
      case 'h':
        height = atoi(optarg);
        break;
      case 'o':
        output_fname = optarg;
        break;
      case 'p':
        parallel_method = atoi(optarg);
        break;
      case '?':
        usage(argv[0]);
        break;
      default:
        usage(argv[0]);
        break;
    }
  }
  
  if (input_fname == 0 || output_fname == 0 || sigma <= 0 || width <= 0 || height <= 0) {
    usage(argv[0]);
  }

  setParallelMethod(parallel_method);

  io_timing = wtime();

  float **objCoords = NULL;
  int numObjs = 0;
  loadtxt(input_fname, objCoords, numObjs);

  io_timing = wtime() - io_timing;

  float **densityMap;
  malloc2D(densityMap, width, height, float);

  kde_timing = wtime();

  kde2D(objCoords, numObjs, width, height, sigma, densityMap);

  kde_timing = wtime() - kde_timing;

  timing = wtime();
  savetxt(output_fname, densityMap, width, height);
  io_timing += wtime() - timing;

  free2D(objCoords);
  free2D(densityMap);
  
  printf("\nPerforming *** Kernel Density Estimation (sequential version) ***\n");

  printf("Input file:          %s\n", input_fname);
  printf("numObjs              = %d\n", numObjs);
  printf("sigma                = %f\n", sigma);
  printf("DensityMap Dimension = (%d, %d)\n", width, height);
  printf("I/O time             = %10.4f sec\n", io_timing);
  printf("Computation timing   = %10.4f sec\n", kde_timing);
  return 0;
}
