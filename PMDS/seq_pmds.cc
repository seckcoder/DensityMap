#include "seq_pmds.h"
// @paramaters
int _numOfObjs;
int _numOfPivots;
int _numOfEdges;

// @Input Data
#define USE

// DeltaMatrix is a numOfObjs * numOfObjs matrix
float **_DeltaMatrix;
float **_DeltaMatrix2;

// CMatrix is a numOfObjs * numOfPivots matrix
float **_CMatrix;
Mat<float> CMatrix;

float **_PivotMatrix;

int *_PivotIndex;

float _maxDis;

/**
	@input file 
	@output **matrix
*/
void loadMatrixFromFile(const char* file){
	FILE * fd = fopen(file,"r");
	_maxDis = 0;
	int dummy;
	fscanf(fd,"%d %d %d\n",&_numOfObjs,&dummy,&_numOfEdges);
	printf("_numOfObjs is %d, _numOfEdges is %d\n",_numOfObjs,_numOfEdges);
	int n1, n2;
	float weight;
	_DeltaMatrix = new float*[_numOfObjs];
	_DeltaMatrix2 = new float*[_numOfObjs];
	float *p = new float[_numOfObjs * _numOfObjs];
	float *p2 = new float[_numOfObjs * _numOfObjs];
	assert(p != NULL);
	assert(p2 != NULL);
	//initilize the _DeltaMatrix, and the DeltaMatrix
	for(int i = 0; i < _numOfObjs;i++){
		_DeltaMatrix[i] = (p + i * _numOfObjs);
		_DeltaMatrix2[i] = (p2 + i *_numOfObjs);
		for(int j = 0; j < _numOfObjs;j++){
			_DeltaMatrix[i][j] = std::numeric_limits<float>::max();
			_DeltaMatrix2[i][j] = std::numeric_limits<float>::max();
		}
	}
	
	for(int i = 0; i < _numOfEdges;i++){
		fscanf(fd,"%d %d %f\n",&n1,&n2,&weight);
		//printf("%d %d %f\n",n1,n2,weight);
		_DeltaMatrix[n1-1][n2-1] = weight;
		_DeltaMatrix2[n1-1][n2-1] = weight * weight;
		//assume overflow is not going to happen
		if(_maxDis < weight)
			_maxDis = weight;
	}
	_maxDis /= 2;
	printf("_maxDis of all weights of edges are %f\n",_maxDis);

	for(int i = 0; i < _numOfObjs;i++){
		for(int j = 0; j < _numOfObjs;j++){
			if(_DeltaMatrix[i][j] == std::numeric_limits<float>::max()){
				_DeltaMatrix[i][j] = _maxDis;
				_DeltaMatrix2[i][j] = _maxDis * _maxDis;
			}
		}
	}

	fclose(fd);
	return ;
}

/**
	@input numOfPivots, deltaMatrix
	@output pivotIndex
*/
void selectPivot(){
	_PivotIndex = new int[_numOfPivots];
	for(int i = 0; i < _numOfPivots;i++){
		_PivotIndex[i] = i;
	}
}



/**
	@input pivotMatrix
	@output cMatrix
*/
void calculateCMatrix(){
	printf("calculating CMatrix\n");
	CMatrix.set_size(_numOfObjs,_numOfPivots);
	float term1, term2, term3, term4;
	for(int i = 0; i < _numOfObjs;i++){
		for(int j = 0; j < _numOfPivots;j++){
			//printf("[%d,%d]\n",i,j);
			float val = 0;
			#ifdef USE
			term1 = _DeltaMatrix[i][j] * _DeltaMatrix[i][j];
			#else
			term1 = _DeltaMatrix2[i][j];
			#endif
			term2 = 0.0f;
			term3 = 0.0f;
			term4 = 0.0f;
			for(int k = 0;k < _numOfObjs;k++){
				//printf("k1 [%d]\n",k);
				#ifdef USE
				term2 += _DeltaMatrix[k][j] * _DeltaMatrix[k][j];
				term3 += _DeltaMatrix[i][k] * _DeltaMatrix[i][k];
				#else
				term2 += _DeltaMatrix2[k][j];
				term3 += _DeltaMatrix2[i][k];
				#endif
				for(int k2 = 0;k2 < _numOfPivots;k2++){
					//printf("k2 [%d]\n",k2);
					#ifdef USE
					term4 += _DeltaMatrix[k][k2] * _DeltaMatrix[k][k2];
					#else
					term4 += _DeltaMatrix2[k][k2];
					#endif
				}
			}
			term2 /= (-_numOfObjs);
			term3 /= (-_numOfPivots);
			term4 /= (_numOfObjs * _numOfPivots);
			CMatrix(i,j) =  (-0.5) * (term1 + term2 + term3 + term4);
		}
	}
	printf("calculating CMatrix done\n");
}



/**
	@input cMatrix
	@output x, y
*/
void powerIterate(){

}













