#include "seq_pmds.h"
// @paramaters
int numOfObjs;
int numOfPivots;
int numOfEdges;

// @Input Data

// DeltaMatrix is a numOfObjs * numOfObjs matrix
float **_DeltaMatrix;
// CMatrix is a numOfObjs * numOfPivots matrix
float **_CMatrix;

float **_PivotMatrix;

int *_PivotIndex;


/**
	@input file 
	@output **matrix
*/
void loadMatrixFromFile(float **matrix, const char* file){
	FILE * fd = fopen(file,"r");
	int dummy;
	fscanf(fd,"%d %d %d\n",&numOfObjs,&dummy,&numOfEdges);
	printf("numOfObjs is %d, numOfEdges is %d\n",numOfObjs,numOfEdges);
	int n1, n2;
	float weight;
	matrix = new float*[numOfObjs];
	float *p = new float[numOfObjs * numOfObjs];
	for(int i = 0; i < numOfObjs;i++){
		matrix[i] = (p + i * numOfObjs);
	}
	for(int i = 0; i < numOfEdges;i++){
		fscanf(fd,"%d %d %f\n",&n1,&n2,&weight);
		matrix[n1-1][n2-1] = weight;
	}
	fclose(fd);
	return ;
}

/**
	@input numOfPivots, deltaMatrix
	@output pivotIndex
*/
void selectPivot(int *pivotIndex,const int &numOfPivots, float **deltaMatrix){

}



/**
	@input pivotMatrix
	@output cMatrix
*/
void calculateCMatrix(float **cMatrix,
						float **pivotMatrix,
						const int &numOfPivots,
						float **deltaMatrix,
						const int &numOfObjs){


}



/**
	@input cMatrix
	@output x, y
*/
void powerIterate(int *x, int *y,float **cMatrix){

}


/**
	@input m1, m2
	@output result
*/
void matrixMul(float **result, 
				float **m1, float **m2, 
				const int &numOfrow, const int &numOfcol){

}












