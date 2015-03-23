#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include "kde.h"

using namespace std;

static void usage(char *argv0) {
  // TODO: usage
  printf("Usage:...\n");
  exit(-1);
}

int main(int argc, char **argv) {
  int opt;
  char *input_fname;
  char *output_fname;
  float sigma = 0;
  double timing, io_timing, kde_timing;
  int width = 0, height = 0;
  while ( (opt=getopt(argc,argv,"i:s:w:h:o:")) != EOF ) {
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

  io_timing = wtime();

  float **objCoords = NULL;
  int numObjs = 0;
  loadtxt(input_fname, objCoords, numObjs);

  io_timing = wtime() - io_timing;

  float **densityMap;
  malloc2D(densityMap, width, height, float);

  kde_timing = wtime();

  kde2D(objCoords, numObjs, densityMap, width, height, sigma);

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
