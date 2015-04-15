#ifndef _GRAPH_H
#define _GRAPH_H

#include "stdio.h"
#include <vector>
#include <string.h>
#include "Node.h"

class Graph
{
private:
	std::vector<Node*> nodeList;
	std::vector<Edge*> edgeList;
	float **disMat;
	int numOfNodes;
public:
	Graph();
	~Graph();
	void initGraph(const int &);
	void connect(const int &id1,const int &id2,const float& weight);
	void dijkstraTree(const int &id);
	void dijkstraTreeAll();
	void showDis(const int &id);
	void loadUnweightedFromFile(const char[]);
};


#endif