
#include "stdio.h"
#include "Graph.h"

int main(int argc, char **argv){
	Graph g;
	g.loadUnweightedFromFile(argv[1]);
	g.dijkstraTreeAll();
	return 0;
}