#include <string>
#include <fstream>
#include <vector>

#include "kde.h"
using std::string;
using std::ifstream;
using std::vector;
using std::ofstream;

void loadtxt(const string &filename, float **&objCoords, int &numObjs) {
  ifstream ifs(filename.c_str(), std::ifstream::in);
  vector<std::pair<float,float> > vecCoords;
  while (true) {
    float x,y;
    ifs >> x >> y;
    if (ifs.eof()) break;
    vecCoords.push_back(std::pair<float,float>(x,y));
  }
  numObjs = vecCoords.size();
  malloc2D(objCoords, numObjs, 2, float);
  for (int i = 0; i < vecCoords.size(); i++) {
    objCoords[i][0] = vecCoords[i].first;
    objCoords[i][1] = vecCoords[i].second;
  }
  ifs.close();
}

void savetxt(const string &filename, float **densityMap, int width, int height) {
  ofstream ofs(filename.c_str(), std::ofstream::out);
  // TODO: save density map
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      float x = float(i) / float(width-1);
      float y = float(j) / float(height-1);
      ofs << x << "\t" << y << "\t" << densityMap[i][j] << std::endl;
    }
  }
  ofs.close();
}
