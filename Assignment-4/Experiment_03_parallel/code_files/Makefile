CXX = g++
CXXFLAGS = -O2 -Wall
OMP_FLAGS = -fopenmp
LDFLAGS = -lm

all: experiment1 experiment2 experiment3

experiment1: experiment1.cpp init.cpp utils.cpp
	$(CXX) $(CXXFLAGS) -o experiment1 experiment1.cpp init.cpp utils.cpp $(LDFLAGS)

experiment2: experiment2.cpp init.cpp utils.cpp
	$(CXX) $(CXXFLAGS) -o experiment2 experiment2.cpp init.cpp utils.cpp $(LDFLAGS)

experiment3: experiment3.cpp init.cpp utils.cpp
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -o experiment3 experiment3.cpp init.cpp utils.cpp $(LDFLAGS)

clean:
	rm -f experiment1 experiment2 experiment3 Mesh.out *.o

.PHONY: all clean
