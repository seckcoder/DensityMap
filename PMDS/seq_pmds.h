#ifndef _SEQ_PMDS_H
#define _SEQ_PMDS_H
#include "stdio.h"


void loadMatrixFromFile(const char *file);
/**
	@input numOfPivots, deltaMatrix
	@output pivotIndex
*/
void selectPivot(int *pivotIndex,const int &numOfPivots, float **deltaMatrix);

/**
	@input pivotMatrix
	@output cMatrix
*/
void calculateCMatrix(float **cMatrix,
						float **pivotMatrix,
						const int &numOfPivots,
						float **deltaMatrix,
						const int &numOfObjs);

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