CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

all: solver

solver: main.cpp black_scholes.cpp
	$(CXX) $(CXXFLAGS) main.cpp black_scholes.cpp -o solver

test: test_solver.cpp black_scholes.cpp
	$(CXX) $(CXXFLAGS) test_solver.cpp black_scholes.cpp -o test_solver
	./test_solver

clean:
	rm -f solver test_solver