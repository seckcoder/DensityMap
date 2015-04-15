CXX=g++-4.9

CFLAGS=-std=c++11
LIBFLAGS=-l armadillo 
PATH_TO_BGL=/Users/Edward/Downloads/boost_1_57_0
INCLUDEFLAGS=-I $(PATH_TO_BGL)
DEBUG=-g

seq_main: seq_pmds.cc seq_pmds.h seq_main.cc
	$(CXX) seq_pmds.cc seq_main.cc -o seq_main $(LIBFLAGS)
Graph:
	$(CXX) -c Graph.cc -o Graph.o $(CFLAGS)
test:
	$(CXX) $(LIBFLAGS) $(INCLUDEFLAGS) test.cc -o test
test_graph: test_graph.cc Graph.cc Graph.h
	$(CXX) test_graph.cc Graph.cc -o test_graph $(DEBUG) $(CFLAGS)

clean:
	rm seq_pmds;
	rm test 
	rm test_graph
	rm *.o