#ifndef _H_KDE
#define _H_KDE

#include <cassert>
#include <string>
#include <cstdio>
#include <cstdlib>

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)


#define free2D(mem) {\
  free(mem[0]); \
  free(mem); \
}

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

double wtime(void);


void loadtxt(const std::string &filename, float **&objCoords, int &numObjs);
void savetxt(const std::string &filename, float **densityMap, int width,
    int height);

void kde2D(
    float **objCoords,
    int numObjs,
    int width,
    int height,
    float sigma,
    float **densityMap);
void setParallelMethod(int method);

#endif
