#include "Graph.h"
#include <unordered_set>
#include <limits>
Graph::Graph(){

}
Graph::~Graph(){
	printf("Graph: delete nodeList\n");
	for(int idx = 0; idx < numOfNodes;idx++){
		delete this->nodeList[idx];
	}
	//free the content
	printf("Graph: delete disMat[0]\n");
	delete disMat[0];
	//free the pointer array
	printf("Graph: delete disMat\n");
	delete disMat;
}

void Graph::initGraph(const int &numOfNodes){
	this->numOfNodes = numOfNodes;
	float *p = new float[this->numOfNodes * this->numOfNodes];
	memset(p,0,sizeof(float) * this->numOfNodes * this->numOfNodes);
	this->disMat = new float*[this->numOfNodes];
	for(int idx = 0; idx < this->numOfNodes;idx++){
		this->disMat[idx] = (p + idx * this->numOfNodes);
		this->nodeList.push_back(new Node(idx));
	}
}


void Graph::connect(const int &id1, const int &id2, const float& weight){
	this->nodeList[id1]->edges.push_back(new Edge(id2,weight));
	this->nodeList[id1]->edges.push_back(new Edge(id1,weight));
	return ;
}

void Graph::loadUnweightedFromFile(const char filepath[]){
	FILE* fd = fopen(filepath,"r");
	int size;
	printf("Start to read from file\n");
	fscanf(fd,"%d\n",&size);
	printf("size is %d\n",size);
	this->initGraph(size);
	int id1, id2;
	while(fscanf(fd,"%d %d\n",&id1,&id2) != EOF){
		this->connect(id1-1,id2-1,float(1));
	}
	printf("data loaded\n");
	fclose(fd);
	return ;
}

void Graph::dijkstraTreeAll(){
	for(int i = 0; i < this->numOfNodes;i++){
		this->dijkstraTree(i);
		this->showDis(i);
	}
}

void Graph::showDis(const int &id){
	for(int idx = 0;idx < this->numOfNodes;idx++){
		if(this->disMat[id][idx] != std::numeric_limits<float>::max())
		printf("[%d] %f, ",idx,this->disMat[id][idx]);
	}
	printf("\n");
}
void Graph::dijkstraTree(const int &id){
	std::unordered_set<int> unvisited;
	for(int idx = 0;idx < this->numOfNodes;idx++){
		unvisited.insert(idx);
		disMat[id][idx] = std::numeric_limits<float>::max();
	}
	disMat[id][id] = 0;
	int curId = id;
	Node* n;
	std::unordered_set<int>::iterator it;
	std::unordered_set<int>::iterator tmp;
	int cnt = 0;
	while(!unvisited.empty()){
		//printf("running curId %d\n",curId);
		cnt++;
		it = unvisited.find(curId);
		if(it == unvisited.end()){
			printf("Graph: dijkstraTree() Error\n");
			return ;
		}
		n = this->nodeList[*it];
		//printf("disMat[%d][%d] is %f\n",id,n->id,disMat[id][n->id]);
		for(int i = 0; i < n->edges.size();i++){
			if(disMat[id][n->id] + n->edges[i]->weight < disMat[id][n->edges[i]->id]){
				disMat[id][n->edges[i]->id] = disMat[id][n->id] + n->edges[i]->weight;
				//printf("disMat[%d][%d] is changed to %f\n",id,n->edges[i]->id,disMat[id][n->edges[i]->id]);
			}
		}
		//printf("all neighbor nodes of %d traversed\n",*it);
		unvisited.erase(it);
		tmp = unvisited.begin();
		if(tmp == unvisited.end())
			break;
		curId = *tmp;
		float minDis = disMat[id][*tmp];
		while(tmp != unvisited.end()){
			//printf("*tmp %d\n",*tmp);
			if(minDis > disMat[id][*tmp]){
				minDis = disMat[id][*tmp];
				curId = *tmp;
			}
			tmp++;
		}
		//break;
	}
	printf("Graph: dijkstraTree for [%d] done, %d nodes visited\n",id,cnt);
	return ;
}