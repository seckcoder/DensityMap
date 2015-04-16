#include "seq_pmds.h"
#include "stdlib.h"

int main(int argc, char **argv){

	loadMatrixFromFile(argv[1]);
	_numOfPivots = atoi(argv[2]);
	calculateCMatrix();
	return 0;
}