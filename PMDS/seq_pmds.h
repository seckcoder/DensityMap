#ifndef _SEQ_PMDS_H
#define _SEQ_PMDS_H
#include "stdio.h"
#include "assert.h"
#include <armadillo>
#include <limits>
// @paramaters
extern int _numOfObjs;
extern int _numOfPivots;
extern int _numOfEdges;

using namespace arma;
// @Input Data

// DeltaMatrix is a numOfObjs * numOfObjs matrix
extern float **_DeltaMatrix;

// CMatrix is a numOfObjs * numOfPivots matrix
extern float **_CMatrix;

extern float **_PivotMatrix;

extern int *_PivotIndex;

extern void loadMatrixFromFile(const char *file);
/**
	@input numOfPivots, deltaMatrix
	@output pivotIndex
*/
extern void selectPivot(int *pivotIndex,const int &numOfPivots, float **deltaMatrix);

/**
	@input pivotMatrix
	@output cMatrix
*/
void calculateCMatrix();

/**
	@input cMatrix
	@output x, y
*/
void powerIterate(int *x, int *y,float **cMatrix);


/**
	@input m1, m2
	@output result
*/
void matrixMul(float **result, float **m1, float **m2, const int &numOfrow, const int &numOfcol);


#endif