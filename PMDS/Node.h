#ifndef _NODE_H
#define _NODE_H
#include <vector>

struct Edge{
	int id;
	float weight;
	Edge(const int &id,const float &weight):id(id),weight(weight){};
};

struct Node{
	int id;
	std::vector<Edge*> edges;
	Node(const int &id):id(id){};
};

#endif